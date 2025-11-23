# ingestion/indexer.py
from ingestion.loader import scan_documents
from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_blocks
from embedding.embedder import Embedder
from storage.milvus_store import MilvusVectorStore
from config.settings import settings
from tqdm import tqdm

# def build_index(source_dir="sourcepdf"):
#     """
#     构建保险知识库索引：
#     1. 扫描所有文件
#     2. 抽取文字 + 表格（带上下文）
#     3. 对文字内容分块
#     4. 嵌入 + 写入 Milvus
#     """
#     print("🚀 开始构建索引 ...")
#     docs = scan_documents(source_dir)
#     if not docs:
#         print("⚠️ 没有找到可索引的文件。")
#         return

#     embedder = Embedder()
#     store = MilvusVectorStore()
#     total_chunks = 0

#     for doc in tqdm(docs, desc="索引进度"):
#         try:
#             parsed_blocks = parse_pdf(doc["path"])
#             # print(f"📄 解析完成：{doc['path']}，提取到 {len(parsed_blocks)} 个内容块。")
#             # print(f"预览内容块：{parsed_blocks[:2]}")  # 打印前两个内容块以供调试
#             if not parsed_blocks:
#                 print(f"⚠️ 文件无有效内容：{doc['path']}")
#                 continue

#             for block in parsed_blocks:
#                 # content = block.get("text", "").strip()
#                 # modality = block.get("modality", "text")
#                 # page = block.get("page", 0)
                

#                 # 跳过空块
#                 if not content:
#                     continue

#                 # 仅文本进行分块；表格保持整块
#                 if modality == "text":
#                     chunks = chunk_blocks(parsed_blocks)
#                 else:
#                     chunks = [content]
                
#                 for c in chunks:
#                     emb = embedder.embed_text([c])[0]
#                     meta = {
#                         **doc["metadata"],
#                         "page": page,
#                         "modality": modality
#                     }
#                     store.add([emb], [Chunk(c, meta)])
#                     total_chunks += 1

#         except Exception as e:
#             print(f"❌ 文件处理失败: {doc['path']} ({e})")

#     print(f"✅ 索引完成，共写入 {total_chunks} 个文本块。")
import numpy as np
import zlib
import json
import base64

def ensure_1d(vec, dim=None):
    if vec is None:
        return None

    # numpy: squeeze to 1D
    if isinstance(vec, np.ndarray):
        vec = vec.reshape(-1,).astype("float32")
        return vec

    # list: flatten ALL nested lists robustly
    if isinstance(vec, list):
        flattened = []

        def _flatten(x):
            if isinstance(x, list):
                for e in x:
                    _flatten(e)
            else:
                flattened.append(float(e))

        _flatten(vec)  # recursive flatten

        vec = np.array(flattened, dtype="float32")

    # fix dimension if provided
    if dim is not None and len(vec) != dim:
        if len(vec) > dim:
            vec = vec[:dim]
        else:
            vec = np.pad(vec, (0, dim - len(vec)))

    return vec



def compress_table_json(table_json: dict) -> str:
    if not table_json:
        return ""
    raw = json.dumps(table_json).encode("utf-8")
    zipped = zlib.compress(raw)
    return base64.b64encode(zipped).decode("utf-8")


def build_index(source_dir="sourcepdf"):

    print("🚀 开始构建 IRAG_MM 多模态索引 ...")

    docs = scan_documents(source_dir)
    if not docs:
        print("⚠️ 没有找到可索引的文件。")
        return

    embedder = Embedder()
    store = MilvusVectorStore()

    total = 0
    batch_records = []
    batch_size = 100


    for doc in tqdm(docs, desc="索引进度"):
        try:
            blocks = parse_pdf(doc["path"])
            if not blocks:
                print(f"⚠️ 无有效内容：{doc['path']}")
                continue

            # 注入 metadata
            for b in blocks:
                b.setdefault("metadata", {})
                b["metadata"].update({
                    "source": doc.get("path", ""),
                    "company": doc.get("company", ""),
                    "category": doc.get("category", ""),
                    "page_number": b["metadata"].get("page_number"),
                    "modality": b.get("modality"),
                })

            # chunk 化文本/表格
            chunks = chunk_blocks(blocks, max_length=500, overlap=50)

            # ------------------------------------------------------
            # 为每个 chunk 构造 record
            # ------------------------------------------------------
            for c in chunks:
                modality = c.get("modality")
                meta = c.get("metadata", {})

                text_value = None
                #table_json = None
                table_blob = None
                text_vec = None
                table_vec = None

                # 文本块
                if modality == "text":
                    raw_text = (c.get("text") or "").strip()
                    if not raw_text:
                        continue

                    text_value = raw_text
                    # embed_text 返回 shape: (1,1024)
                    text_vec = embedder.embed_text([raw_text])[0]

                # 表格块
                elif modality == "table":
                    table = c.get("table")
                    if not table:
                        continue

                    header = table.get("header", [])
                    rows = table.get("rows", [])

                    table_json = table
                    table_blob = compress_table_json(table_json)
                    table_vec = embedder.embed_table(header, rows)

                else:
                    continue

                # 至少要有一个 vector
                if text_vec is None and table_vec is None:
                    continue

                # ----------- 关键：flatten vector -----------------
                if text_vec is not None:
                    print("DEBUG TEXT_VEC:", text_vec, type(text_vec))
                    text_vec = ensure_1d(text_vec, store.text_dim)

                if table_vec is not None:
                    table_vec = ensure_1d(table_vec, store.table_dim)

                batch_records.append({
                    "modality": modality,
                    "text": text_value,
                    #"table_json": table_json,
                    "table_blob": table_blob,
                    "text_vec": text_vec,
                    "table_vec": table_vec,
                    "metadata": meta,
                })


                # 批量写入
                if len(batch_records) >= batch_size:
                    store.add_records(batch_records)
                    total += len(batch_records)
                    batch_records = []

        except Exception as e:
            print(f"❌ 文件失败：{doc['path']} ({e})")

    # 剩余写入
    if batch_records:
        store.add_records(batch_records)
        total += len(batch_records)

    print(f"🎉 多模态索引构建完成，共写入 {total} 个块。")
