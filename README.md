# 📘 LLM Document Assistant

> PDF 문서 검색(RAG) + 회의록 요약 + 음성 STT를 수행하는  
> **로컬 LLM 기반 문서 지능화 시스템**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](#)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](#)

---

## 📋 프로젝트 개요 (Project Overview)

LLM Document Assistant는 **LangChain 기반 RAG(Document QA) 시스템**입니다.

사용자가 업로드한 PDF 문서를 벡터DB에 색인한 뒤 질의응답을 수행하며,  
CSV/PDF 회의록 자동 요약 및 Whisper 기반 음성 → 텍스트 변환(STT)을 통합 제공합니다.

특히 다음 구조를 통해 대형 LLM을 안정적으로 서빙합니다:

- device_map="auto" 기반 멀티 GPU 샤딩
- CPU 오프로딩 차단 (max_memory 설정)
- HuggingFace generate() 직행 구조
- 한국어 출력 후처리 파이프라인

---

## ✨ 주요 기능 (Key Features)

- 🔎 RAG 기반 문서 질의응답
- 🗂 PDF → 청킹 → 임베딩 → Chroma 영구 저장
- 🧠 멀티 GPU 샤딩 기반 LLM 서빙
- 📝 CSV/PDF 회의록 자동 요약 (Markdown 출력)
- 🎙 Whisper 기반 로컬 STT
- 🇰🇷 한국어 출력 품질 보정 (LaTeX 제거, 한자 제거, 문장 정리)

---

## 🧠 색인 → 검색 → 생성 아키텍처

### 1️⃣ 문서 색인 (Indexing)

```
PDF 업로드
→ PyMuPDF / pymupdf4llm 로딩
→ CharacterTextSplitter 청킹
→ HuggingFaceEmbeddings 벡터화
→ Chroma(VectorDB) 영구 저장
```

### 2️⃣ 질의응답 (RAG Flow)

```
User Question
→ Retriever (Top-K 검색)
→ PromptTemplate 구성
→ HF LLM generate()
→ 한국어 후처리
→ Answer 반환
```

### 3️⃣ 회의록 요약

```
CSV / PDF 입력
→ 표준 포맷(start|end|speaker_id|text) 변환
→ LLM 요약 생성
→ Markdown 저장
```

### 4️⃣ 음성 → 텍스트 (STT)

```
Audio 파일
→ faster-whisper
→ 세그먼트 추출
→ CSV 저장
→ UI 미리보기
```

---

## 🛠 기술 스택 (Tech Stack)

| Category | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| LLM | HuggingFace Transformers |
| Embedding | intfloat/multilingual-e5-base |
| RAG | LangChain |
| Vector DB | Chroma |
| UI | Gradio 5.x |
| STT | faster-whisper |
| PDF Loader | PyMuPDF, pymupdf4llm |
| Infra | Multi-GPU (device_map="auto") |

---

## 📁 프로젝트 구조

```
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
```

---

## 🚀 설치 및 실행

### 1️⃣ 설치

```bash
git clone https://github.com/your-repo/llm-document-assistant.git
cd llm-document-assistant
pip install -r requirements.txt
```

### 2️⃣ 실행

```bash
python app.py
```

브라우저 접속:

```
http://localhost:7860
```

---

## 🧪 테스트 실행

```bash
python quick_check.py
```

PDF 색인 → 벡터 생성 → 질의응답 동작 여부 확인

---

## 🐛 설계 포인트

### 🔹 멀티 GPU 안정화

- device_map="auto" 적용
- max_memory 설정으로 CPU 오프로딩 차단
- 임베딩 레이어 디바이스 자동 탐지
- 디바이스 mismatch 오류 방지

### 🔹 한국어 출력 품질 강화

- 유니코드 NFKC 정규화
- LaTeX 수식 자동 치환
- 접속부사 단독 라인 제거
- 한자(CJK) 제거

---

## 👤 담당 역할 (My Contribution)

| 기능 | 기여도 |
|------|--------|
| RAG 아키텍처 설계 및 체인 구성 | 100% |
| 멀티 GPU LLM 샤딩 및 디바이스 안정화 | 100% |
| PDF 다단계 로딩 파이프라인 설계 | 100% |
| 회의록 자동 요약 시스템 구현 | 100% |
| Whisper 기반 STT 통합 | 100% |
| Gradio UI 통합 및 전체 파이프라인 연결 | 100% |
