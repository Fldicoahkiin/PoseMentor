from __future__ import annotations

from dataclasses import dataclass

# COCO-17 关键点顺序，YOLO11-Pose 直接输出该顺序。
JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

JOINT_INDEX = {name: idx for idx, name in enumerate(JOINT_NAMES)}

# 2D/3D 绘制骨架连接。
SKELETON_EDGES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

# 用于动作纠错的关节角定义。
ANGLE_DEFS = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
    "left_shoulder": (11, 5, 7),
    "right_shoulder": (12, 6, 8),
    "left_hip": (5, 11, 13),
    "right_hip": (6, 12, 14),
}


@dataclass(slots=True)
class JointAdviceTemplate:
    joint_name: str
    more_flex_text: str
    less_flex_text: str


JOINT_ADVICE = {
    "left_knee": JointAdviceTemplate("左膝", "左膝角度不够，请再弯一点", "左膝弯曲过多，请稍微伸直"),
    "right_knee": JointAdviceTemplate("右膝", "右膝角度不够，请再弯一点", "右膝弯曲过多，请稍微伸直"),
    "left_elbow": JointAdviceTemplate("左肘", "左肘再收紧一点", "左肘有点过度弯曲，放松一点"),
    "right_elbow": JointAdviceTemplate("右肘", "右肘再收紧一点", "右肘有点过度弯曲，放松一点"),
    "left_shoulder": JointAdviceTemplate("左肩", "左肩抬得不够，再抬高一点", "左肩抬得太高，请下压一点"),
    "right_shoulder": JointAdviceTemplate("右肩", "右肩抬得不够，再抬高一点", "右肩抬得太高，请下压一点"),
    "left_hip": JointAdviceTemplate("左髋", "左髋打开不够，再展开一点", "左髋外展过多，收回来一点"),
    "right_hip": JointAdviceTemplate("右髋", "右髋打开不够，再展开一点", "右髋外展过多，收回来一点"),
}
