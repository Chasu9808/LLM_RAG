# core/ui.py
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────────
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import List

import gradio as gr
import pandas as pd

from config.config import DEBUG_MODE, SERVER_NAME, SERVER_PORT
from core.embeddings import get_retriever, add_documents
from core.loader import load_and_split_local
from core.llm_chain import build_chain
from core.meeting_local import (
    prepare_from_csv,
    prepare_from_pdf,   # ✅ PDF 준비 함수 추가
    summarize_meeting,
)
from core.stt_local import transcribe_to_csv


# ────────────────────────────────────────────────────────────────────────────────
# Runtime defaults
# ────────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _normalize_md(md: str) -> str:
    """Markdown 출력 가독성 보정: 헤더/리스트 줄바꿈 정리."""
    s = (md or "").replace("\r\n", "\n")
    s = re.sub(r"\s*(#{1,6}\s)", r"\n\n\1", s)  # 헤더 앞 공백 줄 보장
    s = re.sub(r"\s*([-*]\s+)", r"\n\1", s)     # 리스트 기호 줄 시작 보정
    return s.strip() + "\n"


# ────────────────────────────────────────────────────────────────────────────────
# Handlers (GUI 탭 순서와 동일)
# ① 색인(Index) → ② 질의(Query) → ③ 회의록/문서 요약 → ④ 음성→텍스트(STT)
# ────────────────────────────────────────────────────────────────────────────────
def _handle_index(files, chain_state):
    """① PDF 색인 후 체인 갱신."""
    if not files:
        return "업로드된 파일이 없습니다.", chain_state, None

    paths: List[str] = [f.name for f in files if hasattr(f, "name")]
    names = ", ".join(os.path.basename(p) for p in paths)

    try:
        # 1) PDF → 문서 청크 생성
        docs = load_and_split_local(paths)

        # 2) 벡터DB(영구 저장소)에 추가
        add_documents(docs)

        # 3) 최신 retriever/chain 갱신
        new_chain = build_chain(get_retriever())

        # 4) 상태 메시지 + 업로더 리셋(None)
        status = f"색인 완료 ✅ : {names}"
        return status, new_chain, None
    except Exception:
        err = "[예외 발생]\n" + traceback.format_exc()
        return err, chain_state, None


def _handle_query(message, history, chain_state, initial_chain):
    """② 질의응답 처리."""
    t0 = time.time()
    history = history or []
    chain = chain_state or initial_chain

    try:
        result = chain.invoke({"question": message})
    except Exception:
        err_txt = "[예외 발생]\n" + traceback.format_exc()
        history.append([message, err_txt])
        return "", history, chain

    answer = (result.get("answer") or "").strip()

    lines = [answer if answer else "응답이 비어있습니다."]
    lines.append(f"⏱ {time.time() - t0:.2f}s")

    history.append([message, "\n".join(lines)])
    return "", history, chain


def _handle_meeting_local(file, sep, map_json, csz):
    """③ CSV/PDF → 준비 → 요약(Markdown) 생성. 미리보기/다운로드 반환."""
    if not file:
        return "CSV/PDF 파일을 업로드하세요.", "", None, None

    in_path = file.name if hasattr(file, "name") else file
    filename = os.path.basename(in_path)
    ext = Path(in_path).suffix.lower()

    # 화자 매핑(JSON)
    try:
        speaker_map = json.loads(map_json) if map_json else None
    except Exception:
        speaker_map = None

    out_dir = "./outputs/meetings"
    try:
        # 확장자에 따라 준비 단계 분기
        if ext == ".pdf":
            prepared_csv = prepare_from_pdf(in_path, out_dir=out_dir)
        else:
            prepared_csv = prepare_from_csv(
                in_path, sep=sep or "|", speaker_map=speaker_map, out_dir=out_dir
            )

        # 요약 생성
        md_path, md_text = summarize_meeting(prepared_csv, out_dir=out_dir, chunk_size=int(csz))
        md_text = _normalize_md(md_text)

        info = (
            f"✅ 문서 준비 완료 : {filename}\n"
            f"- prepared_csv: {prepared_csv}\n"
            "✅ 요약 생성 완료(Markdown)\n"
            f"- summary_md: {md_path}"
        )
        # 안내문, 미리보기 텍스트, 파일 경로(다운로드), 업로드 리셋(None)
        return info, md_text, md_path, None
    except Exception:
        return "[예외 발생]\n" + traceback.format_exc(), "", None, None


def _handle_audio_to_text(file, mname, lang, use_vad):
    """④ 오디오 → Whisper STT(CSV) → 미리보기 텍스트."""
    if not file:
        return "❌ 오디오 파일을 업로드하세요.", "", None

    audio_path = file.name if hasattr(file, "name") else file
    filename = os.path.basename(audio_path)
    lang_arg = None if (lang is None or lang == "auto") else lang

    try:
        csv_path = transcribe_to_csv(
            audio_path,
            csv_out="./outputs/meetings/meeting_stt.csv",
            model_size=mname,
            language=lang_arg,
            vad=bool(use_vad),
        )
    except Exception as e:
        return f"❌ STT 실패: {e}", "", None, None

    try:
        df = pd.read_csv(csv_path, sep="|")
        lines = []
        for _, r in df.iterrows():
            t = str(r.get("text", "")).strip()
            if t:
                lines.append(f"- {t}")
        md_text = "\n".join(lines) if lines else "(변환된 텍스트가 없습니다)"
    except Exception as ex:
        md_text = f"(미리보기 생성 중 오류) {ex}"

    info = f"✅ STT 완료 : {filename}"
    return info, md_text, csv_path, None


# ────────────────────────────────────────────────────────────────────────────────
# UI Entrypoint
# ────────────────────────────────────────────────────────────────────────────────
def launch_ui(initial_chain=None):
    """
    앱 UI 실행 진입점.
    - 1) 색인(Index): PDF 색인 → retriever/chain 갱신
    - 2) 질의(Query): 질문/답변 + 출처
    - 3) 회의록/문서 요약(로컬): CSV/PDF 업로드 → 요약(Markdown) → 미리보기/다운로드
    - 4) 음성→텍스트(로컬): Whisper STT → CSV/미리보기
    """
    if initial_chain is None:
        initial_chain = build_chain(get_retriever())

    with gr.Blocks() as demo:
        gr.Markdown("### 📘 LLM-ChatBot")
        chain_state = gr.State(initial_chain)

        # ① 색인(Index)
        with gr.Tab("1) 색인(Index)"):
            gr.Markdown("업로드한 PDF를 로컬 벡터DB(영구)에 색인합니다.")
            files = gr.File(label="PDF 업로드", file_count="multiple", file_types=[".pdf"])
            btn_index = gr.Button("색인 실행")
            out_index = gr.Textbox(label="결과", lines=3)

            btn_index.click(
                _handle_index,
                inputs=[files, chain_state],
                outputs=[out_index, chain_state, files],
                queue=False,
            )

        # ② 질의(Query)
        with gr.Tab("2) 질의(Query)"):
            chatbot = gr.Chatbot(label="채팅창", height=520, type="tuples")
            msg = gr.Textbox(label="질문을 입력하세요", placeholder="예) 환불 정책 핵심만 알려줘")
            clear = gr.Button("초기화")

            msg.submit(
                lambda m, h, cs: _handle_query(m, h, cs, initial_chain),
                [msg, chatbot, chain_state],
                [msg, chatbot, chain_state],
                queue=False,
            )
            clear.click(lambda: None, None, chatbot, queue=False)

        # ③ 회의록/문서 요약(로컬)
        with gr.Tab("3) 회의록 요약(로컬)"):
            gr.Markdown("CSV **또는** PDF를 업로드하면 **로컬 LLM**으로 요약(Markdown)을 생성합니다.")
            meet_file = gr.File(
                label="회의록/문서 업로드 (CSV | PDF)",
                file_count="single",
                file_types=[".csv", ".pdf"],   # ✅ 확장
            )
            inp_sep = gr.Textbox(label="CSV 구분자", value="|", scale=1)
            inp_map = gr.Textbox(
                label='발화자 매핑(JSON, CSV일 때만 적용) 예: {"SPEAKER_00":"AI","SPEAKER_01":"홍길동"}',
                value='{"SPEAKER_00":"AI","SPEAKER_01":"홍길동"}',
            )
            inp_chunksz = gr.Slider(label="청크 크기(발화 행 수)", minimum=100, maximum=600, value=300, step=50)

            btn_run = gr.Button("요약 실행")

            out_info = gr.Textbox(label="결과 안내", lines=4)
            out_preview = gr.Code(label="요약 미리보기 (Markdown)")
            out_file = gr.File(label="요약 MD 다운로드")

            btn_run.click(
                _handle_meeting_local,
                [meet_file, inp_sep, inp_map, inp_chunksz],
                [out_info, out_preview, out_file, meet_file],
                queue=False,
            )

        # ④ 음성 → 텍스트(로컬)
        with gr.Tab("4) 음성 → 텍스트(로컬)"):
            gr.Markdown("오디오 파일을 업로드하면 로컬 Whisper로 **텍스트로만 변환**합니다. (요약 미수행)")
            aud_file = gr.File(
                label="오디오 업로드",
                file_count="single",
                file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
            )
            dd_model = gr.Dropdown(
                label="Whisper 모델",
                choices=["tiny", "base", "small", "medium", "large-v2"],
                value="base",
            )
            dd_lang = gr.Dropdown(
                label="언어(옵션, 자동감지=auto)",
                choices=["auto", "ko", "en", "ja", "zh"],
                value="ko",
            )
            cb_vad = gr.Checkbox(label="VAD(묵음 기반 분절) 사용", value=True)

            btn_run_a = gr.Button("변환 실행")

            out_info_a = gr.Textbox(label="결과 안내", lines=6)
            out_preview_a = gr.Code(label="STT 미리보기 (텍스트)", language="markdown", interactive=False)
            out_dl_csv = gr.File(label="STT CSV 다운로드")

            btn_run_a.click(
                _handle_audio_to_text,
                [aud_file, dd_model, dd_lang, cb_vad],
                [out_info_a, out_preview_a, out_dl_csv, aud_file],
                queue=False,
            )

    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, debug=DEBUG_MODE)
