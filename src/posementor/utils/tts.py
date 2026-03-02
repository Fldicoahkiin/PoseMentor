from __future__ import annotations

import asyncio
import time
from pathlib import Path

from gtts import gTTS


def _sanitize_filename(text: str) -> str:
    keep = [c for c in text if c.isalnum()]
    return "".join(keep)[:20] or "voice"


async def _edge_tts_async(text: str, output_path: Path, voice: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))


def synthesize_speech(
    text: str,
    output_dir: Path,
    voice_engine: str = "edge_tts",
    edge_voice: str = "zh-CN-XiaoxiaoNeural",
) -> Path | None:
    """返回语音文件路径，失败时返回 None，不中断推理流程。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{int(time.time() * 1000)}_{_sanitize_filename(text)}.mp3"

    try:
        if voice_engine == "edge_tts":
            asyncio.run(_edge_tts_async(text=text, output_path=out_file, voice=edge_voice))
        elif voice_engine == "gtts":
            gTTS(text=text, lang="zh-cn").save(str(out_file))
        else:
            # edge_tts 失败时可以切换到 gTTS；这里保持两种主路径。
            gTTS(text=text, lang="zh-cn").save(str(out_file))
        return out_file
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS] 语音合成失败，已忽略: {exc}")
        return None
