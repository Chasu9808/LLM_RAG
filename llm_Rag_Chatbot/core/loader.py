# core/loader.py
# ─────────────────────────────────────────────────────────────
# PDF 문서 로더 (드롭인 교체 버전)
# - 다양한 로더(PyMuPDF, pymupdf4llm, PyPDF)를 단계적으로 시도
# - 인코딩 깨짐/유니코드 문제를 자동 보정
# - 로드된 Document 객체를 LangChain용으로 정리 후, 청크 단위로 분할
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import List
import unicodedata

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

# ① 기본 로더: PyMuPDFLoader — 한글 추출 성능이 가장 우수, 레이아웃 유지력 좋음
from langchain_community.document_loaders import PyMuPDFLoader
# ② 보조 로더: pymupdf4llm — 전체 PDF를 Markdown 포맷으로 일괄 추출
import pymupdf4llm
# ③ 백업 로더: PyPDFLoader — 상기 로더 실패 시 최종 fallback
from langchain_community.document_loaders import PyPDFLoader

# 텍스트 복원 라이브러리(ftfy)는 선택적으로 사용
try:
    from ftfy import fix_text
    HAS_FTFY = True
except Exception:
    HAS_FTFY = False


# ─────────────────────────────────────────────────────────────
#  텍스트 정리 함수
# ─────────────────────────────────────────────────────────────
def _clean_ko(text: str) -> str:
    """
    - 유니코드 정규화(NFC)
    - ftfy로 깨진 문자열 보정 (옵션)
    - 잘못된 치환문자(�, U+FFFD) 제거
    """
    t = unicodedata.normalize("NFC", text or "")
    if HAS_FTFY:
        t = fix_text(t)
    return t.replace("\uFFFD", "")  # 인코딩 오류로 들어온 치환문자 제거


# ─────────────────────────────────────────────────────────────
#  텍스트 청크 분할 함수
# ─────────────────────────────────────────────────────────────
def _split(documents: List[Document]) -> List[Document]:
    """
    LangChain CharacterTextSplitter를 이용해
    문서(Document 리스트)를 chunk_size / chunk_overlap 단위로 분리한다.
    """
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────────────────────
#  로더별 PDF 페이지 로딩 함수
# ─────────────────────────────────────────────────────────────
def _load_pages_with_pymupdf(path: str) -> List[Document]:
    """
    ① PyMuPDFLoader 기반 로드
    - PDF 페이지 단위로 텍스트를 추출
    - 각 페이지를 Document 객체로 반환
    - 유니코드 보정(_clean_ko) 적용
    """
    docs = PyMuPDFLoader(path).load()
    return [Document(page_content=_clean_ko(d.page_content), metadata=d.metadata) for d in docs]


def _load_pages_with_pymupdf4llm(path: str) -> List[Document]:
    """
    ② pymupdf4llm 기반 로드
    - PDF 전체를 Markdown 형식으로 추출 (페이지 구분 없음)
    - 긴 문서 요약용 등에서 레이아웃 정보 유지에 유리
    """
    md = pymupdf4llm.to_markdown(path)
    md = _clean_ko(md)
    return [Document(page_content=md, metadata={"source": path, "format": "md"})]


def _load_pages_with_pypdf(path: str) -> List[Document]:
    """
    ③ PyPDFLoader 기반 로드 (최후 fallback)
    - PyMuPDF 계열 실패 시 사용
    - 단순 텍스트 추출이지만 호환성은 높음
    """
    docs = PyPDFLoader(path).load()
    return [Document(page_content=_clean_ko(d.page_content), metadata=d.metadata) for d in docs]


# ─────────────────────────────────────────────────────────────
#  다단계 로더 (Best Effort 방식)
# ─────────────────────────────────────────────────────────────
def _load_pages_best_effort(path: str) -> List[Document]:
    """
    다양한 PDF 로더를 순차적으로 시도:
      ① PyMuPDF → ② pymupdf4llm → ③ (선택적으로) OCR → ④ PyPDF

    - PyMuPDF 결과에 깨짐문자(�)가 10개 이상 있으면 pymupdf4llm로 재시도
    - 모든 시도 실패 시 PyPDFLoader로 최종 처리
    """
    try:
        docs = _load_pages_with_pymupdf(path)
        # 텍스트 깨짐(�)이 일정 이상이면 pymupdf4llm 재시도
        if sum(d.page_content.count("\uFFFD") for d in docs) > 10:
            docs = _load_pages_with_pymupdf4llm(path)
        return docs
    except Exception:
        try:
            return _load_pages_with_pymupdf4llm(path)
        except Exception:
            try:
                # (선택) OCR 로드가 구현된 경우 자동 호출
                return _load_pages_with_ocr(path)
            except Exception:
                return _load_pages_with_pypdf(path)


# ─────────────────────────────────────────────────────────────
#  메인 엔트리 함수
# ─────────────────────────────────────────────────────────────
def load_and_split_local(file_paths: List[str]) -> List[Document]:
    """
    파일 목록(file_paths)을 입력 받아 다음을 수행:
      1) 각 파일을 best-effort 방식으로 로드
      2) 문서를 모두 병합
      3) CharacterTextSplitter로 청크 단위 분할
      4) 최종 Document 리스트 반환

    반환: 분할된 Document 리스트
    """
    all_docs: List[Document] = []
    for p in file_paths:
        docs = _load_pages_best_effort(p)
        all_docs.extend(docs)

    # 청크 단위로 분리
    texts = _split(all_docs)
    print(f"✅(LOCAL) 파일 {len(file_paths)}개, 원문 {len(all_docs)}개, 분할 청크 {len(texts)}개")
    return texts
