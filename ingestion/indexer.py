from ingestion.loader import scan_documents
from ingestion.parser import parse_pdf
from ingestion.chunker import chunk_blocks
from embedding.embedder import Embedder
from storage.milvus_store import MilvusVectorStore
import numpy as np
import zlib
import json
import base64


# ====================== 全局安全截断 ======================
def safe_truncate(text: str, max_len: int = 2000) -> str:
    if not text:
        return ""
    return text.strip()[:max_len]


def safe_truncate_table(text: str) -> str:
    return safe_truncate(text, max_len=1000)


def ensure_1d(vec, dim=None):
    if vec is None:
        return None
    if isinstance(vec, np.ndarray):
        vec = vec.reshape(-1, ).astype("float32")
        return vec
    if isinstance(vec, list):
        flattened = []

        def _flatten(x):
            if isinstance(x, list):
                for e in x:
                    _flatten(e)
            else:
                flattened.append(float(e))

        _flatten(vec)
        vec = np.array(flattened, dtype="float32")
    if dim is not None and len(vec) != dim:
        if len(vec) > dim:
            vec = vec[:dim]
        else:
            vec = np.pad(vec, (0, dim - len(vec)))
    return vec


def compress_table_json(table_json: dict) -> str:
    """增强版：确保返回非空，且打印调试信息"""
    if not table_json or "header" not in table_json or "rows" not in table_json:
        print(f"[WARNING] 无效表格数据，跳过: {table_json}")
        return ""
    try:
        raw = json.dumps(table_json, ensure_ascii=False).encode("utf-8")
        zipped = zlib.compress(raw)
        return base64.b64encode(zipped).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] 表格压缩失败: {e}")
        return ""


# ====================== 表格三路拆分（修复版） ======================
def split_table_three_way(table: dict):
    if not table or "header" not in table or "rows" not in table:
        return []
    header = table.get("header", [])
    rows = table.get("rows", [])
    chunks = []

    # 1. 表级模态
    chunks.append({
        "modality": "table",
        "table": table,
        "text": safe_truncate_table(f"表格：{' | '.join(header[:10])}")
    })

    # 2. 列级模态（只保留核心列）
    core_columns = ["金额", "赔付", "保额", "保费", "免赔额", "报销比例", "最高", "最低"]
    for col_idx, col_name in enumerate(header):
        if col_idx >= 5 and not any(k in col_name for k in core_columns):
            continue
        col_values = [str(row[col_idx]) for row in rows[:10] if len(row) > col_idx]
        if len(rows) > 10:
            col_values.append(f"...共{len(rows)}行")
        chunks.append({
            "modality": "column",
            "table": table,
            "text": safe_truncate_table(f"{col_name}：{', '.join(col_values)}")
        })

    # 3. 行级模态（只保留前5行）
    for row_idx, row in enumerate(rows[:5]):
        row_content = []
        for col_idx, cell in enumerate(row[:8]):
            if col_idx < len(header):
                row_content.append(f"{header[col_idx]}:{cell}")
        if len(row) > 8:
            row_content.append("...")
        chunks.append({
            "modality": "row",
            "table": table,
            "text": safe_truncate_table(f"行{row_idx + 1}：{' | '.join(row_content)}")
        })

    return chunks


# ====================== 索引构建（最终修复版） ======================
def build_index(source_dir="sourcepdf"):
    print("🚀 开始构建最终版索引（含表格三路+table_blob校验）...")

    docs = scan_documents(source_dir)
    if not docs:
        print("⚠️ 没有找到可索引的文件。")
        return

    embedder = Embedder()
    store = MilvusVectorStore()

    total = 0
    valid_table_count = 0
    batch_records = []
    batch_size = 100

    for doc_idx, doc in enumerate(docs):
        print(f"[进度] 处理第 {doc_idx + 1}/{len(docs)} 个文件: {doc['path']}")

        try:
            blocks = parse_pdf(doc["path"])
            if not blocks:
                continue

            for b in blocks:
                b.setdefault("metadata", {})
                b["metadata"].update({
                    "source": doc.get("path", ""),
                    "company": doc.get("company", ""),
                    "category": doc.get("category", ""),
                    "page_number": b["metadata"].get("page_number"),
                    "modality": b.get("modality"),
                })

            chunks = chunk_blocks(blocks, max_length=500, overlap=50)

            for c in chunks:
                modality = c.get("modality")
                meta = c.get("metadata", {}).copy()

                # ---------------- 文本块 ----------------
                if modality == "text":
                    raw_text = (c.get("text") or "").strip()
                    if not raw_text:
                        continue
                    text_value = safe_truncate(raw_text)
                    text_vec = embedder.embed_text([text_value])[0]
                    text_vec = ensure_1d(text_vec, store.vector_dim)

                    batch_records.append({
                        "modality": modality,
                        "text": text_value,
                        "table_blob": "",
                        "vector": text_vec,
                        "metadata": meta,
                    })

                # ---------------- 表格块（含三路+非空校验） ----------------
                elif modality == "table":
                    table = c.get("table")
                    if not table:
                        continue

                    three_way_chunks = split_table_three_way(table)

                    for tw_chunk in three_way_chunks:
                        sub_modality = tw_chunk["modality"]
                        sub_text = tw_chunk["text"]
                        sub_table = tw_chunk["table"]
                        table_blob = compress_table_json(sub_table)

                        # 【关键】跳过table_blob为空的无效记录
                        if not table_blob:
                            continue

                        if sub_modality == "table":
                            vec = embedder.embed_table(sub_table["header"][:10], sub_table["rows"][:5])
                        else:
                            vec = embedder.embed_text([sub_text])[0]
                        vec = ensure_1d(vec, store.vector_dim)

                        sub_meta = meta.copy()
                        sub_meta["sub_modality"] = sub_modality

                        batch_records.append({
                            "modality": sub_modality,
                            "text": sub_text,
                            "table_blob": table_blob,
                            "vector": vec,
                            "metadata": sub_meta,
                        })
                        valid_table_count += 1

                # 批量入库
                if len(batch_records) >= batch_size:
                    store.add_records(batch_records)
                    total += len(batch_records)
                    batch_records = []
                    print(f"  → 已入库 {total} 个块 | 有效表格: {valid_table_count} 个")

        except Exception as e:
            print(f"❌ 文件失败：{doc['path']}，错误：{str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 剩余数据入库
    if batch_records:
        store.add_records(batch_records)
        total += len(batch_records)

    print(f"\n🎉 索引构建完成！")
    print(f"   总入库数量: {total}")
    print(f"   有效表格数量: {valid_table_count}")
    print(f"   文本数量: {total - valid_table_count}")


if __name__ == "__main__":
    build_index()