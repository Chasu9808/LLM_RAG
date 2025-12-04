# quick_check.py
from core.sqlite_patch import patch_sqlite
patch_sqlite()
from core.loader import load_and_split_pdf
from core.embeddings import build_vectorstore
from core.llm_chain import build_chain

texts = load_and_split_pdf()
retriever = build_vectorstore(texts)
chain = build_chain(retriever)
print(chain.invoke({"question": "인공지능으로 인한 이슈는?"}))