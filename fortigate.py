# ==============================
# 一鍵式 FortiGate PDF → JSON/CSV → Chroma 向量庫
# ==============================

import os
import json
import csv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# --------------------------
# 使用者設定
# --------------------------
PDF_FOLDER = r"C:\FortiDocs"  # 放 PDF 的資料夾
CHUNK_SIZE = 300  # 每段字數
JSON_OUT = "fortigate_chunks.json"
CSV_OUT = "fortigate_chunks.csv"
CHROMA_COLLECTION_NAME = "fortigate_docs"

# --------------------------
# 1. 讀取 PDF
# --------------------------
all_chunks = []

for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        reader = PdfReader(pdf_path)
        print(f"正在處理: {filename} ...")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                words = text.split()
                for j in range(0, len(words), CHUNK_SIZE):
                    chunk_text = " ".join(words[j:j+CHUNK_SIZE])
                    if chunk_text.strip():
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "pdf": filename,
                                "page": i+1
                            }
                        })

# --------------------------
# 2. 儲存 JSON
# --------------------------
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
print(f"完成 JSON：{JSON_OUT}")

# --------------------------
# 3. 儲存 CSV
# --------------------------
with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "pdf", "page"])
    writer.writeheader()
    for c in all_chunks:
        writer.writerow({
            "text": c["text"],
            "pdf": c["metadata"]["pdf"],
            "page": c["metadata"]["page"]
        })
print(f"完成 CSV：{CSV_OUT}")

# --------------------------
# 4. Embeddings + Chroma
# --------------------------
print("開始生成向量 embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = [model.encode(c["text"]).tolist() for c in tqdm(all_chunks)]

client = chromadb.Client()
collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
collection.add(
    documents=[c["text"] for c in all_chunks],
    metadatas=[c["metadata"] for c in all_chunks],
    embeddings=embeddings
)
print(f"完成向量資料庫：{CHROMA_COLLECTION_NAME}")
