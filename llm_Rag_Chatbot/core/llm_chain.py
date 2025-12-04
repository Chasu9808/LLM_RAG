# core/llm_chain.py
import os
import time
import re
import torch
import unicodedata
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from config.config import MODEL_NAME, ATTN_IMPL, MAX_NEW_TOKENS, MULTI_GPU_SHARDING

# ─────────────────────────────────────────────────────────────
# CUDA / PyTorch 가속 관련 기본 설정
# - 큰 텐서 할당 안정화, TF32 허용, matmul 정밀도 튜닝
# ─────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def _load_llm():
    """
    LLM과 토크나이저를 로드하고, 단일/멀티 GPU에 맞게 배치한다.
    - MULTI_GPU_SHARDING=True면 device_map="auto" 샤딩과 max_memory로 CPU 오프로딩을 차단
    - False면 단일 GPU(cuda:0)에 전부 올림
    반환: (tokenizer, model, embed_device_str)
    """
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # 사용할 dtype 결정 (bf16 지원되면 bf16, 아니면 fp16)
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    if MULTI_GPU_SHARDING:
        # ✅ 멀티 GPU 샤딩(PCIe) + CPU 오프로딩 차단
        max_memory = {}
        n = torch.cuda.device_count()
        for i in range(n):
            max_memory[i] = "78GiB"  # 각 GPU에 허용할 최대 메모리(환경에 맞게 조절)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",          # 하위 모듈을 여러 GPU로 분산(샤딩)
            torch_dtype=torch_dtype,    # ❗ dtype → torch_dtype 로 변경
            low_cpu_mem_usage=True,
            attn_implementation=ATTN_IMPL,  # "sdpa" 또는 "flash_attention_2" 등
            max_memory=max_memory,          # CPU로의 암묵적 오프로딩 방지
            offload_folder=None,            # 디스크 오프로딩 방지
        )
        embed_dev = _pick_embed_device(model)  # 임베딩 레이어의 실제 디바이스를 찾아 입력 텐서 디바이스를 맞춤
    else:
        # ✅ 단일 GPU(cuda:0) 고정
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=None,          # 단일 장치에 수동 배치
            torch_dtype=torch_dtype,  # ❗ 여기서도 torch_dtype 사용
            low_cpu_mem_usage=True,
            attn_implementation=ATTN_IMPL,
        ).to("cuda:0")
        embed_dev = "cuda:0"

    # 생성 관련 세이프가드 설정
    model.eval()
    model.generation_config.pad_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.use_cache = True

    return tok, model, embed_dev


def _pick_embed_device(model) -> str:
    """
    샤딩된 모델에서 임베딩 레이어가 올려진 실제 디바이스를 추출한다.
    - 입력 텐서를 같은 디바이스로 보낸 뒤 generate() 호출 시 디바이스 mismatch를 방지
    """
    dev = "cuda:0"
    try:
        dm = getattr(model, "hf_device_map", None)
        if isinstance(dm, dict):
            v = dm.get("model.embed_tokens", next(iter(dm.values())))
            if isinstance(v, str) and v.startswith("cuda"):
                dev = v
            else:
                # 예: 'cuda:1' 같은 문자열을 파싱
                dev = f"cuda:{int(str(v).split(':')[-1])}"
    except Exception:
        pass
    return dev


def _format_docs(docs):
    """
    검색된 문서 리스트(docs)의 page_content를 이어붙여 컨텍스트 문자열로 만든다.
    """
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))


# ─────────────────────────────────────────────────────────────
# 한국어 후처리 관련 정규식 및 토큰 셋
# - 원형 숫자/불릿/LaTeX 수식/다중 공백 제거 등
# ─────────────────────────────────────────────────────────────
_CIRCLED_NUM = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
_BAD_TOKENS = {"또", "또,", "과", "그리고", "그리고,", "또한", "또한,"}
_LATEX = re.compile(r"(\$[^$]+\$|\\\([^\)]+\\\)|\\\[.*?\\\])", re.DOTALL)
_BULLET = re.compile(r"^[\s•\-·*○●◦■□★☆・]+")
_MULTI_WS = re.compile(r"\s{2,}")  # ← 다중 공백 축소용

def _to_nfkc(s: str) -> str:
    """유니코드 NFKC 정규화(전각/호환문자 → 보통문자 등)."""
    return unicodedata.normalize("NFKC", s or "")

def _strip_latex(s: str) -> str:
    """LaTeX/수식 토큰을 '(수식 생략)'으로 치환하여 원형 노출을 방지."""
    return _LATEX.sub("(수식 생략)", s)

def _clean_ko_lines(s: str) -> str:
    """
    줄 단위로 불릿/접속부사 단독 라인 제거 → 한 줄로 합침 → 공백/마침표 보강.
    """
    lines = []
    for l in s.splitlines():
        l = _BULLET.sub("", l).strip()  # 앞쪽 불릿/특수문자 제거
        if not l:
            continue
        if l in {"또", "또,", "그리고", "그리고,", "또한", "또한,"}:  # 접속부사 단독 라인 제거
            continue
        lines.append(l)

    s = " ".join(lines)
    s = _MULTI_WS.sub(" ", s).strip()  # 다중 공백 축소
    # 마지막에 완결 부호가 없다면 마침표 보강(경우에 따라 취향 조절)
    if s and not re.search(r"[.?!다]$", s):
        s += "."
    return s

def _postprocess_answer(raw: str) -> str:
    """
    모델 출력 후처리 파이프라인:
    - 유니코드 정규화 → LaTeX 치환 → 일부 숫자+명사 붙임 보정 → 한국어 라인 정리
    """
    s = _to_nfkc(raw)
    s = _strip_latex(s)
    s = re.sub(r"(\d+)\s*(문서|특성|형식)", r"\1 \2", s)  # 숫자-명사 붙음 보정 예시
    s = _clean_ko_lines(s)
    return s


def build_chain(retriever):
    """
    LangChain Runnable을 생성한다.
    - RAG 흐름: retriever로 컨텍스트 검색 → 프롬프트 구성 → HF 모델로 generate → 후처리 → 답변/출처 반환
    """
    tok, model, embed_dev = _load_llm()

    # 프롬프트: 한국어 문장형 강제 + LaTeX 금지 + 단락형 요약 지시 + 근거부재시 대응
    qa_prompt = PromptTemplate(
        template=(
            "[문서 기반 질의응답]\n"
            "역할: 너는 한글 설명 전용 도우미다.\n"
            "규칙:\n"
            "1) 출력은 **반드시 자연스러운 한국어 문장형**으로만 작성한다. (영어, 로마자, 원문 인용 금지)\n"
            "2) 문서에 수식/LaTeX가 있어도 **우리말로 풀어서** 설명한다. `$...$`, `\\frac{{}}` 등 원형을 그대로 쓰지 않는다.\n"
            "3) 불릿/목록 대신 1~2개 **완결 문단**으로 요약한다. (끊긴 어구/접속부사 단독 금지)\n"
            "4) 근거가 명확하지 않으면 '문서에 해당 내용이 없습니다.'라고 답한다.\n\n"
            "[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]"
        ),
        input_variables=["context", "question"],
    )

    def _qa_runner(inp: dict) -> dict:
        """
        RunnableLambda 실행 본체:
        1) retriever로 관련 문서 검색
        2) 프롬프트 포맷팅
        3) HF 모델 generate
        4) 한국어 후처리
        5) 답변/출처 반환
        """
        try:
            q = (inp or {}).get("question", "").strip()
            t0 = time.time()

            # 1) 검색
            try:
                docs = retriever.invoke(q)  # LangChain 0.3 Runnable 인터페이스
            except Exception:
                docs = retriever._get_relevant_documents(q)  # 구버전 호환 백업
            t1 = time.time()

            # 2) 프롬프트 구성
            ctx = _format_docs(docs)
            prompt_text = qa_prompt.format(context=ctx, question=q)

            # 3) generate (AMP로 연산 가속)
            with torch.inference_mode(), torch.cuda.amp.autocast(
                dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            ):
                inputs = tok(prompt_text, return_tensors="pt")
                # 샤딩 환경에서는 임베딩 레이어가 올려진 디바이스로 입력 텐서를 이동
                inputs = {k: v.to(embed_dev) for k, v in inputs.items()}
                g0 = time.time()
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,            # 그리디/빔 계열로 일관성 확보
                    temperature=0.2,            # 일부 모델에서 로짓 처리에 영향(안정성)
                    top_p=0.9,
                    repetition_penalty=1.1,     # 반복 억제
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                )
                g1 = time.time()

            # 4) 디코드 및 프롬프트 에코 제거
            text = tok.decode(out_ids[0], skip_special_tokens=True)
            raw_answer = text[len(prompt_text):].strip() if text.startswith(prompt_text) else text.strip()

            # 🔸 후처리(한국어 클린업)
            # NOTE: 현재 파일엔 _clean_ko가 정의되어 있지 않고, _postprocess_answer가 준비되어 있음.
            # 아래 한 줄은 _postprocess_answer로 바꾸는 것이 일관성에 맞음.
            # answer = _clean_ko(raw_answer)
            answer = _postprocess_answer(raw_answer)

            # 5) 성능 로그 + 반환
            total = g1 - t0
            print(f"[perf] retrieve={t1-t0:.2f}s, generate={g1-g0:.2f}s, total={total:.2f}s")
            return {"answer": answer, "source_documents": docs}
        except Exception:
            import traceback
            return {"answer": "[체인 내부 예외]\n" + traceback.format_exc(), "source_documents": []}

    # LangChain Runnable로 체인 구성
    chain = RunnableLambda(_qa_runner)
    print("✅ LLM 체인 초기화 완료 (샤딩/오프로딩 방지 + generate 직행 + 한국어 클린업)")
    return chain
