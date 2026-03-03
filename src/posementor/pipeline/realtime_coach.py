from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from posementor.models.lift_net import PoseLiftTransformer
from posementor.utils.io import ensure_dir
from posementor.utils.kalman import KeypointKalmanSmoother
from posementor.utils.math3d import center_pose
from posementor.utils.scoring import ScoreDetail, dtw_align_indices, evaluate_aligned_sequence
from posementor.utils.tts import synthesize_speech
from posementor.utils.visualize import build_3d_skeleton_figure, draw_metrics_panel, draw_pose_2d


@dataclass(slots=True)
class CoachConfig:
    yolo_weights: str = "yolo11m-pose.pt"
    lift_checkpoint: str = "artifacts/lift_demo.ckpt"
    norm_file: str = "artifacts/lift_demo_norm.npz"
    template_dir: str = "data/processed/aistpp/gt3d"
    seq_len: int = 81
    device: str = "cuda"
    det_conf: float = 0.35
    joint_bad_threshold_mm: float = 60.0
    tts_engine: str = "edge_tts"


class TemplateLibrary:
    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self.templates: dict[str, list[np.ndarray]] = {}
        self._load()

    def _style_from_seq_id(self, seq_id: str) -> str:
        return seq_id.split("_")[0]

    def _load(self) -> None:
        if not self.template_dir.exists():
            return

        for npz_file in sorted(self.template_dir.glob("*.npz")):
            with np.load(npz_file) as data:
                joints = data["joints3d"].astype(np.float32)
                if "style" in data.files:
                    style_val = data["style"]
                    style = str(style_val.item() if np.ndim(style_val) == 0 else style_val)
                else:
                    style = self._style_from_seq_id(npz_file.stem)
            joints = center_pose(joints)
            self.templates.setdefault(style, []).append(joints)

    def available_styles(self) -> list[str]:
        return sorted(self.templates.keys())

    def get_template(self, style: str, min_len: int) -> np.ndarray:
        candidates = self.templates.get(style)
        if not candidates:
            # 若指定舞种缺失，退化为全局最长序列。
            all_seq = [seq for values in self.templates.values() for seq in values]
            if not all_seq:
                raise RuntimeError("模板库为空，请先执行 download_and_prepare_aist.py")
            candidates = all_seq

        best = max(candidates, key=lambda x: x.shape[0])
        if best.shape[0] >= min_len:
            return best[:min_len]

        # 模板帧数不足时重复补齐，保证 DTW 最低可运行。
        repeat = int(np.ceil(min_len / best.shape[0]))
        tiled = np.tile(best, (repeat, 1, 1))
        return tiled[:min_len]


def build_lift_model_from_state(state_dict: dict[str, torch.Tensor]) -> tuple[PoseLiftTransformer, int]:
    max_seq_len = int(state_dict["time_pos_embed"].shape[1]) if "time_pos_embed" in state_dict else 243
    hidden_dim = int(state_dict["time_pos_embed"].shape[2]) if "time_pos_embed" in state_dict else 256
    num_joints = int(state_dict["head.weight"].shape[0] // 3) if "head.weight" in state_dict else 17
    in_dim = (
        int(state_dict["input_proj.weight"].shape[1] // max(num_joints, 1))
        if "input_proj.weight" in state_dict
        else 2
    )
    model = PoseLiftTransformer(
        num_joints=num_joints,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
    )
    return model, max_seq_len


class RealtimeDanceCoach:
    def __init__(self, config: CoachConfig) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        self.detector = YOLO(config.yolo_weights)
        self.model, self.norm_mean, self.norm_std = self._load_lift_model(
            ckpt_path=Path(config.lift_checkpoint),
            norm_file=Path(config.norm_file),
        )
        self.model.to(self.device).eval()

        self.templates = TemplateLibrary(Path(config.template_dir))

        self.smoother = KeypointKalmanSmoother(num_joints=17)
        self.history_2d: deque[np.ndarray] = deque(maxlen=config.seq_len)

        self.last_advice_text = ""
        self.last_tts_time = 0.0
        self.tts_cooldown_sec = 1.8
        self.voice_dir = ensure_dir(Path("outputs") / "voice")

    def _load_lift_model(
        self,
        ckpt_path: Path,
        norm_file: Path,
    ) -> tuple[PoseLiftTransformer, np.ndarray, np.ndarray]:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"找不到 3D lift 权重: {ckpt_path}，请先运行 train_3d_lift_demo.py"
            )

        state = torch.load(ckpt_path, map_location="cpu")
        cleaned: dict[str, torch.Tensor]
        if isinstance(state, dict) and "state_dict" in state:
            cleaned = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
        else:
            cleaned = state

        model, max_seq_len = build_lift_model_from_state(cleaned)
        model.load_state_dict(cleaned, strict=False)

        if self.config.seq_len > max_seq_len:
            self.config.seq_len = max_seq_len

        if not norm_file.exists():
            mean = np.zeros((1, 1, 2), dtype=np.float32)
            std = np.ones((1, 1, 2), dtype=np.float32)
            return model, mean, std

        with np.load(norm_file) as data:
            mean = data["mean_2d"].astype(np.float32)
            std = data["std_2d"].astype(np.float32)
        std = np.clip(std, 1e-6, None)
        return model, mean, std

    def reset(self) -> None:
        self.history_2d.clear()
        self.smoother.reset()
        self.last_advice_text = ""
        self.last_tts_time = 0.0

    def available_styles(self) -> list[str]:
        return self.templates.available_styles()

    def _extract_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        results = self.detector.predict(frame_bgr, conf=self.config.det_conf, verbose=False)
        if not results:
            return None

        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        keypoints_xy = result.keypoints.xy.cpu().numpy()  # [P,17,2]
        conf = result.keypoints.conf.cpu().numpy()  # [P,17]

        # 选择平均置信度最高的人，满足单人教学场景。
        person_idx = int(np.argmax(conf.mean(axis=1)))
        kp = np.concatenate([keypoints_xy[person_idx], conf[person_idx, :, None]], axis=-1)
        return kp.astype(np.float32)

    def _infer_3d(self) -> np.ndarray | None:
        if len(self.history_2d) < self.config.seq_len:
            return None

        window = np.stack(self.history_2d, axis=0)  # [T,17,3]
        x = window[:, :, :2]
        x = (x - self.norm_mean) / self.norm_std

        x_tensor = torch.from_numpy(x[None]).float().to(self.device)
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()[0]

        pred = center_pose(pred)
        return pred.astype(np.float32)

    def _score_current_window(self, pred3d: np.ndarray, style: str) -> tuple[ScoreDetail, list[str], np.ndarray]:
        ref3d = self.templates.get_template(style=style, min_len=len(pred3d))

        q_idx, r_idx, _ = dtw_align_indices(pred3d, ref3d)
        aligned_pred = pred3d[q_idx]
        aligned_ref = ref3d[r_idx]
        detail = evaluate_aligned_sequence(aligned_pred, aligned_ref)

        bad_joints = [
            joint
            for joint, err in detail.joint_errors_mm.items()
            if err >= self.config.joint_bad_threshold_mm
        ]

        # 返回最后一帧模板姿态用于 3D 可视化。
        ref_last = aligned_ref[-1]
        return detail, bad_joints, ref_last

    def _maybe_tts(self, advice_text: str, score: float) -> Path | None:
        if self.config.tts_engine == "none":
            return None

        now = time.time()
        if score > 90:
            return None
        if advice_text == self.last_advice_text and (now - self.last_tts_time) < self.tts_cooldown_sec:
            return None
        voice_path = synthesize_speech(
            text=advice_text,
            output_dir=self.voice_dir,
            voice_engine=self.config.tts_engine,
        )
        if voice_path is not None:
            self.last_advice_text = advice_text
            self.last_tts_time = now
        return voice_path

    def process_frame(self, frame_rgb: np.ndarray, style: str) -> dict[str, object]:
        """单帧推理接口，供 Gradio 流式回调使用。"""
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        keypoints = self._extract_keypoints(frame_bgr)

        if keypoints is None:
            placeholder = frame_rgb.copy()
            return {
                "annotated": placeholder,
                "score": 0.0,
                "mpjpe_mm": 0.0,
                "angle_error_deg": 0.0,
                "is_ready": False,
                "advice": "未检测到人体，请调整镜头。",
                "bad_joints": [],
                "plot": build_3d_skeleton_figure(np.zeros((17, 3), dtype=np.float32)),
                "audio": None,
            }

        keypoints = self.smoother(keypoints)
        self.history_2d.append(keypoints)

        pred3d = self._infer_3d()
        if pred3d is None:
            vis = draw_pose_2d(frame_bgr, keypoints, bad_joint_names=[])
            vis = draw_metrics_panel(
                vis,
                score=0.0,
                mpjpe_mm=0.0,
                angle_error=0.0,
                advice=f"缓冲中 {len(self.history_2d)}/{self.config.seq_len} 帧",
            )
            vis_rgb = vis[:, :, ::-1]
            return {
                "annotated": vis_rgb,
                "score": 0.0,
                "mpjpe_mm": 0.0,
                "angle_error_deg": 0.0,
                "is_ready": False,
                "advice": "正在收集时序信息，请继续动作。",
                "bad_joints": [],
                "plot": build_3d_skeleton_figure(np.zeros((17, 3), dtype=np.float32)),
                "audio": None,
            }

        detail, bad_joints, ref_last = self._score_current_window(pred3d=pred3d, style=style)
        vis = draw_pose_2d(frame_bgr, keypoints, bad_joint_names=bad_joints)
        vis = draw_metrics_panel(
            vis,
            score=detail.score,
            mpjpe_mm=detail.mpjpe_mm,
            angle_error=detail.angle_error_deg,
            advice=detail.advice_text,
        )
        vis_rgb = vis[:, :, ::-1]
        voice_path = self._maybe_tts(advice_text=detail.advice_text, score=detail.score)

        return {
            "annotated": vis_rgb,
            "score": detail.score,
            "mpjpe_mm": detail.mpjpe_mm,
            "angle_error_deg": detail.angle_error_deg,
            "is_ready": True,
            "advice": detail.advice_text,
            "bad_joints": bad_joints,
            "plot": build_3d_skeleton_figure(pred3d[-1], ref_last),
            "audio": str(voice_path) if voice_path is not None else None,
        }
