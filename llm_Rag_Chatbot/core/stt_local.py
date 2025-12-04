# core/stt_local.py
from __future__ import annotations
import os
import pandas as pd

def _pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", ("float16", "int8_float16")[0]  # 기본: fp16
    except Exception:
        pass
    return "cpu", "int8"  # CPU 기본: int8

def transcribe_to_csv(
    audio_path: str,
    csv_out: str = "./outputs/meetings/meeting_stt.csv",
    model_size: str = "base",              # tiny/small/base/medium/large-v2 ...
    language: str | None = "ko",           # None이면 자동감지
    vad: bool = True
) -> str:
    """
    로컬 faster-whisper로 STT 수행 → CSV 저장
    CSV 스키마: start|end|speaker_id|text (구분자: |)
    """
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    device, compute_type = _pick_device()

    # lazy import (미사용 환경에서 불필요한 로드 방지)
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5
    )

    rows = []
    for seg in segments:
        # diarization(화자분리) 없이 단일 화자로 처리
        rows.append([seg.start, seg.end, "SPEAKER_00", seg.text.strip()])

    df = pd.DataFrame(rows, columns=["start", "end", "speaker_id", "text"])
    df.to_csv(csv_out, sep="|", index=False, encoding="utf-8-sig")
    return csv_out
