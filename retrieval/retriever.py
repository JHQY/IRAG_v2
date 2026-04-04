"""
Multi-modal retriever for IRAG.
"""

import base64
import hashlib
import json
import zlib

from embedding.embedder import Embedder
from retrieval.reranker import Reranker
from storage.milvus_store import MilvusVectorStore


class RAGInterface:
    def __init__(
        self,
        w_text: float = 1.0,
        w_table: float = 1.0,
        gamma: float = 0.7,
        candidate_multiplier: int = 3,
    ):
        print("Initializing multi-modal RAG interface...")
        self.embedder = Embedder()
        self.store = MilvusVectorStore()
        self.reranker = Reranker()

        self.w_text = w_text
        self.w_table = w_table
        self.gamma = gamma
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5, filters: dict = None):
        del filters  # Filters are not wired into Milvus search yet.

        if not query:
            return []

        qtype = self.classify_query(query)
        k_each = max(top_k * self.candidate_multiplier, top_k)

        if qtype == "column":
            q_vec = self.embedder.embed_text([query])[0]
            hits = self.store.search(q_vec, modality="column", top_k=k_each)
            weight = self.w_table
        elif qtype == "row":
            q_vec = self.embedder.embed_text([query])[0]
            hits = self.store.search(q_vec, modality="row", top_k=k_each)
            weight = self.w_table
        elif qtype == "table":
            q_vec = self.embedder.embed_query_table(query)
            hits = self.store.search(q_vec, modality="table", top_k=k_each)
            weight = self.w_table
        else:
            q_vec = self.embedder.embed_text([query])[0]
            hits = self.store.search(q_vec, modality="text", top_k=k_each)
            weight = self.w_text

        if not hits:
            q_vec = self.embedder.embed_text([query])[0]
            hits = self.store.search(q_vec, modality="text", top_k=k_each)
            weight = self.w_text

        fusion_map = {}
        for rank, hit in enumerate(hits, start=1):
            ent = hit.entity
            meta = ent.get("metadata") or {}
            item = {
                "text": ent.get("text"),
                "table": self._decompress_table(ent.get("table_blob")),
                "metadata": meta,
                "modality": ent.get("modality", ""),
            }
            doc_id = self._build_hit_id(item)
            if doc_id not in fusion_map:
                fusion_map[doc_id] = {
                    "fusion_score": 0.0,
                    "item": item,
                }
            fusion_map[doc_id]["fusion_score"] += weight * (1.0 / rank)

        if not fusion_map:
            return []

        fused_items = list(fusion_map.values())
        fused_items.sort(key=lambda x: x["fusion_score"], reverse=True)
        fused_items = fused_items[: top_k * self.candidate_multiplier]

        candidate_texts = [self._build_rerank_text(fi["item"]) for fi in fused_items]
        rerank_scores = self.reranker.rerank(query, candidate_texts)

        f_scores = [fi["fusion_score"] for fi in fused_items]
        f_max, f_min = max(f_scores), min(f_scores)
        r_max, r_min = max(rerank_scores), min(rerank_scores)

        final = []
        for fi, fs, rs in zip(fused_items, f_scores, rerank_scores):
            f_norm = (fs - f_min) / (f_max - f_min) if f_max > f_min else 0.5
            r_norm = (rs - r_min) / (r_max - r_min) if r_max > r_min else 0.5
            score = self.gamma * r_norm + (1 - self.gamma) * f_norm

            final.append(
                {
                    "text": fi["item"]["text"],
                    "table": fi["item"]["table"],
                    "metadata": fi["item"]["metadata"],
                    "modality": fi["item"]["modality"],
                    "score": round(float(1 - score), 4),
                }
            )

        final.sort(key=lambda x: x["score"])
        return final[:top_k]

    def _build_rerank_text(self, item: dict) -> str:
        text = item.get("text") or ""
        table = item.get("table") or {}
        if text:
            return text
        if table:
            return json.dumps(table, ensure_ascii=False)
        return ""

    def _decompress_table(self, blob: str):
        if not blob:
            return {}
        try:
            data = base64.b64decode(blob)
            raw = zlib.decompress(data)
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _build_hit_id(self, item: dict) -> str:
        meta = item.get("metadata") or {}
        payload = self._build_rerank_text(item)
        payload_hash = hashlib.md5(payload.encode("utf-8")).hexdigest() if payload else "empty"
        return "|".join(
            [
                str(meta.get("source", "x")),
                f"p{meta.get('page_number', '0')}",
                str(item.get("modality", "")),
                payload_hash,
            ]
        )

    def _build_context_text(self, item: dict) -> str:
        parts = []
        # 先加文本内容（如果有）
        text = item.get("text") or ""
        if text:
            parts.append(text)

        # 再加表格内容（如果有）—— 这是原来缺失的关键部分
        table = item.get("table") or {}
        if table:
            header = [str(cell) for cell in table.get("header", [])]
            rows = table.get("rows", [])
            if header and rows:
                parts.append("\n【表格数据】")
                parts.append(" | ".join(header))
                parts.append(" | ".join(["---"] * len(header)))
                for row in rows:
                    parts.append(" | ".join(str(cell) for cell in row))

        return "\n".join(parts)

    def retrieve_context(self, query: str, top_k: int = 5):
        hits = self.retrieve(query, top_k=top_k)
        context_parts = []
        for hit in hits:
            context_text = self._build_context_text(hit)
            if context_text:
                context_parts.append(context_text)
        return "\n---\n".join(context_parts)

    @staticmethod
    def classify_query(query: str):
        q = query.lower()

        column_keywords = [
            "average",
            "mean",
            "sum",
            "max",
            "min",
            "highest",
            "lowest",
            "avg",
            "total",
            "maximum",
            "minimum",
            "\u5e73\u5747",
            "\u603b\u8ba1",
            "\u5408\u8ba1",
            "\u6700\u9ad8",
            "\u6700\u4f4e",
            "\u6700\u5927",
            "\u6700\u5c0f",
        ]
        row_keywords = [
            "who",
            "which",
            "whose",
            "\u54ea\u4e00\u4e2a",
            "\u54ea\u4e2a",
            "\u54ea\u79cd",
            "\u54ea\u7c7b",
            "\u54ea\u9879",
        ]
        table_keywords = [
            "table",
            "tabular",
            "\u8868\u683c",
            "\u8868\u4e2d",
            "\u8868\u91cc",
            "\u6570\u636e\u8868",
            "\u8d39\u7387\u8868",
            "\u5bf9\u7167\u8868",
        ]

        if any(k in q for k in column_keywords):
            return "column"
        if any(k in q for k in row_keywords):
            return "row"
        if any(k in q for k in table_keywords):
            return "table"
        return "text"


if __name__ == "__main__":
    rag = RAGInterface()
    q = "AIA\u610f\u5916\u9669\u7684\u8d54\u4ed8\u8303\u56f4\u662f\u4ec0\u4e48\uff1f"
    res = rag.retrieve(q, top_k=3)
    for r in res:
        print(">>> TEXT:", r["text"])
        print(">>> TABLE STRUCT:", r["table"])
        print(">>> META:", r["metadata"])
