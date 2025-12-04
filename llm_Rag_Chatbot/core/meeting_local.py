# core/meeting_local.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import regex as re
import unicodedata

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config.config import MODEL_NAME, MAX_NEW_TOKENS, ATTN_IMPL

# ───────────────────────────────────────────────────────────────────
# Optional PDF deps (graceful fallback)
# ───────────────────────────────────────────────────────────────────
try:
    import pymupdf4llm  # best quality: PDF -> Markdown
    HAS_PYMUPDF4LLM = True
except Exception:
    HAS_PYMUPDF4LLM = False

try:
    import fitz  # PyMuPDF: fallback text extractor
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# 한자(중/일 문자) 검출 → 요약 결과의 비한글 CJK 제거에 사용
_HAN = re.compile(r'[\p{Script=Han}]')


# ───────────────────────────────────────────────────────────────────
# CSV 준비
# ───────────────────────────────────────────────────────────────────
def _read_csv_safely(path: str, sep: str = "|") -> pd.DataFrame:
    """UTF-8-SIG 우선, 실패 시 CP949로 재시도."""
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, sep=sep, encoding="cp949")


def prepare_from_csv(
    csv_path: str,
    sep: str = "|",
    speaker_map: Optional[dict] = None,
    out_dir: str = "./outputs/meetings",
) -> str:
    """회의록 CSV를 표준 포맷(start|end|speaker_id|text)으로 정리하여 저장."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _read_csv_safely(csv_path, sep=sep)

    # 컬럼 표준화 (존재할 경우 rename)
    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("speaker", "speakerid", "speaker_id"):
            rename_map[c] = "speaker_id"
        elif lc in ("content", "utterance", "transcript", "text"):
            rename_map[c] = "text"
        elif lc in ("start_time", "start"):
            rename_map[c] = "start"
        elif lc in ("end_time", "end"):
            rename_map[c] = "end"
    if rename_map:
        df = df.rename(columns=rename_map)

    # 필수 컬럼 보강
    if "speaker_id" not in df.columns:
        df["speaker_id"] = "UNK"
    if "text" not in df.columns:
        raise ValueError("CSV에 text(또는 content/utterance/transcript) 컬럼이 필요합니다.")
    if "start" not in df.columns:
        df["start"] = range(len(df))
    if "end" not in df.columns:
        df["end"] = df["start"]

    # 화자 매핑 적용
    if speaker_map:
        df["speaker_id"] = df["speaker_id"].map(lambda x: speaker_map.get(str(x), str(x)))

    # 안전한 UTF-8-SIG로 저장
    prep = out / f"{Path(csv_path).stem}_prepared.csv"
    df[["start", "end", "speaker_id", "text"]].to_csv(prep, sep="|", index=False, encoding="utf-8-sig")
    return str(prep)


# ───────────────────────────────────────────────────────────────────
# PDF 준비 (신규)
# ───────────────────────────────────────────────────────────────────
def _pdf_to_markdown_or_text(pdf_path: str) -> str:
    """가능하면 Markdown(pymupdf4llm), 아니면 PyMuPDF 텍스트 추출로 대체."""
    if HAS_PYMUPDF4LLM:
        return pymupdf4llm.to_markdown(pdf_path)

    if not HAS_PYMUPDF:
        raise RuntimeError(
            "PDF 추출을 위해 pymupdf4llm 또는 PyMuPDF(fitz)가 필요합니다. "
            "pip install pymupdf4llm pymupdf"
        )

    # Fallback: PyMuPDF 텍스트
    doc = fitz.open(pdf_path)
    parts: List[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(parts)


def prepare_from_pdf(
    pdf_path: str,
    out_dir: str = "./outputs/meetings",
) -> str:
    """
    PDF → 의사-회의록 CSV (start|end|speaker_id|text) 변환.
    speaker_id는 일괄 'DOC'로 지정하여 summarize_meeting() 재사용.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = Path(pdf_path).stem
    csv_out = out / f"meeting_prepared_{base}.csv"

    # 1) PDF → 텍스트/마크다운
    raw = _pdf_to_markdown_or_text(pdf_path)
    raw = unicodedata.normalize("NFKC", raw or "").replace("\r\n", "\n")

    # 2) 문단 단위 분리
    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]

    # 3) 회의록 표준 포맷으로 생성
    rows = []
    for i, p in enumerate(paras):
        rows.append({"start": i, "end": i, "speaker_id": "DOC", "text": p})

    if not rows:
        # 페이지 단위라도 보강
        rows = [{"start": 0, "end": 0, "speaker_id": "DOC", "text": raw or ""}]

    pd.DataFrame(rows).to_csv(csv_out, sep="|", index=False, encoding="utf-8-sig")
    return str(csv_out)


# ───────────────────────────────────────────────────────────────────
# 요약 (기존)
# ───────────────────────────────────────────────────────────────────
def _build_summarizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPL,
        torch_dtype="auto",
    )
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tok.eos_token_id,
    )
    return gen, tok


def _force_korean(s: str) -> str:
    """요약 결과에서 한자 등 비한글 CJK 제거 + 공백 정리."""
    s = unicodedata.normalize("NFKC", s or "")
    s = _HAN.sub("", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def summarize_meeting(
    prepared_csv: str,
    out_dir: str = "./outputs/meetings",
    chunk_size: int = 300,
) -> Tuple[str, str]:
    """
    표준 포맷 CSV(start|end|speaker_id|text)를 받아 한국어 요약 생성.
    (긴 텍스트는 외부에서 chunking하여 준비 CSV를 만들었다고 가정)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _read_csv_safely(prepared_csv, sep="|")

    # 회의록 본문 구성
    lines = []
    for _, r in df.iterrows():
        spk = str(r.get("speaker_id", "")).strip()
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        lines.append(f"{spk}: {txt}")
    whole = "\n".join(lines)

    gen, tok = _build_summarizer()

    sys_prompt = (
        "너는 회의 요약 도우미다. 반드시 **한국어(한글)** 로만 답한다. "
        "영어·중국어·한자 사용 금지. 인명/지명/조직도 한글 표기(예: 朴志恩 X → 박지은). "
        "아래 회의록을 간결한 한국어 문장으로 요약하라.\n\n"
        "요약 포맷:\n"
        "## 개요\n"
        "## 결정사항(Decision)\n"
        "## 액션아이템(Owner/Deadline)\n"
        "## 향후/메모\n"
    )

    prompt = f"{sys_prompt}\n[회의록]\n{whole}\n\n[요약 시작]"

    out_text = gen(prompt)[0]["generated_text"]
    out_text = _force_korean(out_text)

    # 파일 저장(UTF-8-SIG)
    md_path = out / f"{Path(prepared_csv).stem}_summary.md"
    md_path.write_text(out_text, encoding="utf-8-sig")
    return str(md_path), out_text
