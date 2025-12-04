from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config.config import EMBEDDING_MODEL, TOP_K, PERSIST_DIR, COLLECTION_NAME

# ─────────────────────────────────────────────────────────────
#  전역 Singleton 관리용 변수
#   - embeddings 모델과 vectorstore를 전역 1회만 로드하기 위함
#   - 여러 모듈에서 import되어도 동일한 인스턴스를 재사용
# ─────────────────────────────────────────────────────────────
_embeddings = None
_vector_store = None


# ─────────────────────────────────────────────────────────────
#  내부 함수: 임베딩 모델 로더
#   - HuggingFaceEmbeddings()는 문서를 벡터로 변환하는 객체
#   - 전역 변수 _embeddings 에 캐싱하여, 여러 번 초기화하지 않음
# ─────────────────────────────────────────────────────────────
def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        # 최초 1회만 모델 로드 (예: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


# ─────────────────────────────────────────────────────────────
#  내부 함수: Chroma VectorStore 로더
#   - Chroma는 문서 임베딩을 영구적으로 저장/조회하는 벡터DB
#   - persist_directory 로 지정된 경로에 실제 데이터 저장
#   - collection_name 으로 특정 문서그룹 관리 가능
# ─────────────────────────────────────────────────────────────
def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,          # DB 내 컬렉션 이름 (예: "docs")
            persist_directory=PERSIST_DIR,            # 로컬 저장 디렉토리 (예: "./store")
            embedding_function=_get_embeddings(),     # 임베딩 함수 연결
        )
    return _vector_store


# ─────────────────────────────────────────────────────────────
#  문서 추가 함수
#   - 새로운 Document 객체 리스트를 벡터DB에 추가
#   - add_documents() → persist() 호출로 즉시 디스크 반영
#   - 반환값: 추가된 문서(청크)의 개수
# ─────────────────────────────────────────────────────────────
def add_documents(texts: List[Document]) -> int:
    """
    문서(Document 리스트)를 영구 VectorStore에 추가하고 persist.
    반환값: 추가된 청크 수
    """
    vs = _get_vector_store()   # Chroma 인스턴스 획득
    vs.add_documents(texts)    # 새 문서 벡터화 후 추가
    vs.persist()               # 디스크에 영구 저장
    return len(texts)


# ─────────────────────────────────────────────────────────────
#   검색기(retriever) 반환 함수
#   - RAG에서 검색 단계에 사용되는 Retriever 객체를 생성
#   - TOP_K 개수만큼 유사도가 높은 문서를 반환하도록 설정
#   - 이 retriever는 llm_chain.py 에서 build_chain() 시 연결됨
# ─────────────────────────────────────────────────────────────
def get_retriever():
    """
    현재 저장소 상태로 retriever 반환
    """
    vs = _get_vector_store()
    return vs.as_retriever(search_kwargs={"k": TOP_K})
