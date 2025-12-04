# ================================================================
# config/config.py — 전역 환경 설정
# ================================================================

#  모델 설정

MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# 문서 관련 설정
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# 벡터 검색 설정
TOP_K = 2

# 추론/성능 설정
MAX_NEW_TOKENS = 192
ATTN_IMPL = "sdpa"            # flash-attn 사용 시 "flash_attention_2"
MULTI_GPU_SHARDING = True     # True: 2GPU 샤딩, False: 단일 GPU 고정

# 서버 설정
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860
DEBUG_MODE = True

# 벡터DB 영구 저장소
PERSIST_DIR = "./store"       # Chroma 영구 디렉터리
COLLECTION_NAME = "pdf_rag"
