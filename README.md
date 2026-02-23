📘 LLM Document Assistant

PDF 문서 검색(RAG) + 회의록 요약 + 음성 STT를 수행하는 로컬 LLM 기반 문서 지능화 시스템






📋 프로젝트 개요 (Project Overview)

LLM Document Assistant는 LangChain 기반 RAG(Document QA) 시스템으로,
사용자가 업로드한 PDF 문서를 벡터DB에 색인한 뒤 질의응답을 수행합니다.

또한 본 프로젝트는 CSV/PDF 회의록을 입력받아 로컬 LLM 기반 자동 요약을 수행하고,
Whisper(STT)를 활용해 음성 → 텍스트 변환까지 통합한 문서 지능화 파이프라인을 제공합니다.

특히 본 시스템은 멀티 GPU 샤딩(device_map="auto") + CPU 오프로딩 차단 + generate 직행 구조를 적용하여
대형 LLM을 안정적으로 서빙하도록 설계된 것이 핵심 설계 포인트입니다.

✨ 주요 기능 (Key Features)

RAG 기반 문서 질의응답
PDF 업로드 → 청킹 → 임베딩 → Chroma 영구 저장 → LLM generate

멀티 GPU 샤딩 구조
device_map="auto" 기반 모델 분산 + CPU 오프로딩 차단

회의록 자동 요약 시스템
CSV / PDF 입력 → 표준 포맷 변환 → Markdown 요약 생성

로컬 STT 통합
faster-whisper 기반 음성 → 텍스트 변환 (CSV 저장)

한국어 출력 안정화 후처리
LaTeX 제거, 한자 제거, 문장 완결성 보정

🧠 색인 – 검색 – 생성 아키텍처 (Index → Retrieve → Generate)
1️⃣ 문서 색인 (Indexing)
PDF 업로드
→ PyMuPDF / pymupdf4llm 로딩
→ CharacterTextSplitter 청킹
→ HuggingFaceEmbeddings 벡터화
→ Chroma(VectorDB) 영구 저장
2️⃣ 벡터 검색 (Retrieval)
User Question
→ Retriever (Top-K 검색)
→ PromptTemplate 구성
→ HF LLM generate()
→ 한국어 후처리
→ Answer 반환
3️⃣ 회의록 요약 (Meeting Summarization)
CSV / PDF
→ 표준 포맷(start|end|speaker_id|text) 변환
→ LLM 요약 파이프라인
→ Markdown 출력
→ 파일 저장
4️⃣ 음성 → 텍스트 (STT)
Audio
→ faster-whisper 로컬 모델
→ 세그먼트 추출
→ CSV 저장
→ UI 미리보기
🛠️ 기술 스택 (Tech Stack)
Category	Technology
Language	Python 3.10+
LLM	HuggingFace Transformers
Embedding	intfloat/multilingual-e5-base
RAG Framework	LangChain
Vector DB	Chroma
UI	Gradio 5.x
STT	faster-whisper
PDF Loader	PyMuPDF, pymupdf4llm
Infra	Multi-GPU (device_map="auto")
📁 프로젝트 구조 (Project Structure)
LLM-Document-Assistant/
├── app.py
├── quick_check.py
│
├── config/
│   └── config.py
│
├── core/
│   ├── embeddings.py
│   ├── loader.py
│   ├── llm_chain.py
│   ├── meeting_local.py
│   ├── stt_local.py
│   ├── sqlite_patch.py
│   └── ui.py
│
└── requirements.txt
🚀 설치 및 실행 (Installation & Run)
1️⃣ 사전 요구사항

Python 3.10+

CUDA 환경 (권장)

고성능 GPU 환경 권장 (멀티 GPU 지원)

2️⃣ 설치
git clone https://github.com/your-repo/llm-document-assistant.git
cd llm-document-assistant

python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
3️⃣ 실행
python app.py

브라우저 접속:

http://localhost:7860
🧪 테스트 실행 (Optional: Quick Check)
python quick_check.py

PDF 색인 → 벡터 생성 → 질의응답 동작 여부 확인

🐛 설계 포인트 (Design Notes)
🔹 멀티 GPU 안정화

device_map="auto" 적용

max_memory 설정으로 CPU 오프로딩 차단

임베딩 레이어 디바이스 자동 탐지 후 입력 텐서 이동

디바이스 mismatch 오류 방지

🔹 한국어 출력 품질 강화

유니코드 NFKC 정규화

LaTeX 수식 자동 치환

접속부사 단독 라인 제거

한자(CJK) 제거

🔹 PDF 로딩 안정성 확보

PyMuPDF → pymupdf4llm → PyPDFLoader 다단계 fallback 구조

텍스트 깨짐 자동 보정

👤 담당 역할 (My Contribution)
기능	기여도
RAG 아키텍처 설계 및 체인 구성	100%
멀티 GPU LLM 샤딩 및 디바이스 안정화 구현	100%
PDF 다단계 로딩 파이프라인 설계	100%
회의록 CSV/PDF 자동 요약 시스템 구현	100%
Whisper 기반 로컬 STT 통합	100%
Gradio UI 통합 및 전체 파이프라인 연결	100%
