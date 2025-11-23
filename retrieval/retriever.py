# '''Retriever placeholder'''
# """
# RAG 查询接口模块
# 为问答同学提供：简单调用即可使用向量知识检索。
# """

# from embedding.embedder import Embedder
# from storage.milvus_store import MilvusVectorStore
# import numpy as np


# class RAGInterface:
#     """面向问答模块的 RAG 接口封装"""

#     def __init__(self):
#         print("🔗 初始化 RAG 接口组件...")
#         self.embedder = Embedder()
#         self.store = MilvusVectorStore()

#     # ------------------------------------------------------
#     # 基础搜索接口
#     # ------------------------------------------------------
#     def retrieve(self, query: str, top_k: int = 5, filters: dict = None):
#         """
#         输入查询语句 -> 输出最相似文本块及元信息。
#         参数：
#           query: str —— 问题文本
#           top_k: int —— 返回前多少条相似内容
#           filters: dict —— 可选过滤条件，如 {"company": "AIA"}
#         返回：
#           List[{"text": str, "score": float, "metadata": dict}]
#         """
#         # 1️⃣ 嵌入 query
#         try:
#             q_emb = self.embedder.embed_query(query)
#         except Exception as e:
#             print(f"❌ 查询嵌入失败: {e}")  
#             raw_emb = self.embedder.model.encode([query], convert_to_numpy=True, show_progress_bar=False)   
#             q_emb = np.array(raw_emb, dtype=np.float32)[0]
        
#         if isinstance(q_emb, np.ndarray) and q_emb.ndim >1:
#             q_emb = q_emb[0]
       
#         # 2️⃣ 相似检索
#         hits = self.store.similarity_search(q_emb, top_k=top_k, filters=filters)

#         # 3️⃣ 结构化输出
#         results = []
#         for chunk, score in hits:
#             results.append({
#                 "text": chunk.text,
#                 "score": round(score, 4),
#                 "metadata": chunk.metadata
#             })

#         return results

#     # ------------------------------------------------------
#     # 高级接口（预留给 LLM 使用）
#     # ------------------------------------------------------
#     def retrieve_context(self, query: str, top_k: int = 5):
#         """
#         返回一个合并后的上下文字符串，可直接送入 LLM。
#         """
#         hits = self.retrieve(query, top_k=top_k)
#         context = "\n---\n".join([f"{h['text']}" for h in hits])
#         return context


# # -------------------------
# # 调试入口（可独立运行）
# # -------------------------
# if __name__ == "__main__":
#     rag = RAGInterface()
#     query = "怕出意外应该买哪个保险？"
#     results = rag.retrieve(query, top_k=3)
#     print("\n🔍 Top-3 结果:")
#     for i, r in enumerate(results, 1):
#         print(f"\n{i}. [score={r['score']}]")
#         print(r["text"][:400], "...")
#         print("metadata:", r["metadata"])

"""
多模态（文本 + 表格）RAG 接口
- 文本检索：text_vector (bge-m3)
- 表格检索：table_vector (TAPAS + 表格增强文本)
- RAG-Fusion：基于 reciprocal rank 的加权融合
- Cross-Encoder Re-ranking：BAAI/bge-reranker-base

外部调用保持不变：
    rag = RAGInterface()
    rag.retrieve("换地板可以赔多少？")
    rag.retrieve_context("换地板可以赔多少？")
"""

"""
多模态（文本 + 表格）RAG 接口
- 文本检索：text_vector (bge-m3)
- 表格检索：table_vector (TAPAS)
- RAG-Fusion：基于 reciprocal rank 的加权融合
- Cross-Encoder Re-ranking：BAAI/bge-reranker-base
"""

from embedding.embedder import Embedder
from storage.milvus_store import MilvusVectorStore
from retrieval.reranker import Reranker
import zlib
import json
import base64

class RAGInterface:
    def __init__(
        self,
        w_text: float = 1.0,     # 文本检索权重
        w_table: float = 1.0,    # 表格检索权重
        gamma: float = 0.7,      # reranker 在最终融合中的权重
        candidate_multiplier: int = 3,   # 先取多少候选再精排
    ):
        print("🔗 初始化多模态 RAG 接口组件...")
        self.embedder = Embedder()
        self.store = MilvusVectorStore()
        self.reranker = Reranker()

        self.w_text = w_text
        self.w_table = w_table
        self.gamma = gamma
        self.candidate_multiplier = candidate_multiplier
    



    # ------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5, filters: dict = None):

        def decompress_table_blob(blob: str) -> dict:
            if not blob:
                return {}
            data = base64.b64decode(blob)
            raw = zlib.decompress(data)
            return json.loads(raw.decode("utf-8"))

        if not query or not isinstance(query, str):
            return []

        # 1️⃣ Query → 文本 embedding
        try:
            q_vec_text = self.embedder.embed_text([query])[0]
        except Exception as e:
            print(f"❌ 文本 embedding 失败: {e}")
            return []

        # 2️⃣ Query → 表格 embedding（关键步骤）
        try:
            q_vec_table = self.embedder.embed_query_table(query)
        except Exception as e:
            print(f"⚠️ 表格 embedding 失败，fallback 文本模式: {e}")
            q_vec_table = q_vec_text

        # ---------------------------
        # 多路检索
        # ---------------------------
        k_each = max(top_k * self.candidate_multiplier, top_k)

        # 文本通道
        text_hits = self.store.search_text(q_vec_text, top_k=k_each)

        # 表格通道（现在使用 TAPAS embedding）
        table_hits = self.store.search_table(q_vec_table, top_k=k_each)

        if not text_hits and not table_hits:
            return []

        # ------------------------------------------------------
        # 3️⃣ RAG-Fusion
        # ------------------------------------------------------
        fusion_map = {}

        def make_doc_id(ent):
            meta = ent.get("metadata") or {}
            source = meta.get("source", "unknown")
            page = meta.get("page_number", "na")

            # 两模态版本：以 PDF + 页码 为唯一 ID
            # → 能把 table-hit 和 同页的 text-hit 自动融合
            return f"{source}|p{page}"

        def _add_hits(hits, modality_label, weight):
            for rank, hit in enumerate(hits, start=1):

                ent = hit.entity
                doc_id = make_doc_id(ent)

                if doc_id not in fusion_map:
                    fusion_map[doc_id] = {
                        "fusion_score": 0.0,
                        "item": {
                            "modality": modality_label,
                            "text": ent.get("text"),
                            "table": decompress_table_blob(ent.get("table_blob")),
                            "metadata": ent.get("metadata"),
                        }
                    }

                fusion_map[doc_id]["fusion_score"] += weight * (1.0 / rank)

        # ---- 多模态加入 fusion ----
        _add_hits(text_hits, "text", self.w_text)
        _add_hits(table_hits, "table", self.w_table)

        _add_hits(text_hits,  "text",  self.w_text)
        _add_hits(table_hits, "table", self.w_table)

        if not fusion_map:
            return []

        # ------------------------------------------------------
        # 4️⃣ topN 候选（先按 fusion_score 排序）
        # ------------------------------------------------------
        fused_items = list(fusion_map.values())
        fused_items.sort(key=lambda x: x["fusion_score"], reverse=True)

        candidate_count = min(len(fused_items), max(top_k * self.candidate_multiplier, top_k))
        fused_items = fused_items[:candidate_count]

        candidate_texts = [fi["item"]["text"] or "" for fi in fused_items]

        # ------------------------------------------------------
        # 5️⃣ reranker 精排
        # ------------------------------------------------------
        rerank_scores = self.reranker.rerank(query, candidate_texts)

        fusion_scores = [fi["fusion_score"] for fi in fused_items]
        f_max, f_min = max(fusion_scores), min(fusion_scores)
        r_max, r_min = max(rerank_scores), min(rerank_scores)

        def _norm(x, lo, hi):
            if hi <= lo:
                return 0.5
            return (x - lo) / (hi - lo)

        final_items = []
        for fi, f_s, r_s in zip(fused_items, fusion_scores, rerank_scores):
            f_norm = _norm(f_s, f_min, f_max)
            r_norm = _norm(r_s, r_min, r_max)
            relevance = self.gamma * r_norm + (1 - self.gamma) * f_norm
            cost = 1.0 - relevance

            item = fi["item"]
            final_items.append({
                "text": item["text"],
                "table": item["table"],
                "metadata": item["metadata"],
                "modality": item["modality"],
                "score": cost,
            })

        # ------------------------------------------------------
        # 6️⃣ 最终排序 + top_k
        # ------------------------------------------------------
        final_items.sort(key=lambda x: x["score"])
        final_items = final_items[:top_k]

        # 输出格式保持和旧版一致
        return [
            {
                "text": it["text"],
                "table": it["table"],
                "score": round(float(it["score"]), 4),
                "metadata": it["metadata"],
            }
            for it in final_items
        ]


    # ------------------------------------------------------
    # 上下文拼接接口
    # ------------------------------------------------------
    def retrieve_context(self, query: str, top_k: int = 5):
        hits = self.retrieve(query, top_k=top_k)
        return "\n---\n".join([h["text"] for h in hits if h["text"]])


# -------------------------
# 自测
# -------------------------
if __name__ == "__main__":
    rag = RAGInterface()
    q = "AIA意外险的赔付范围是什么？"
    res = rag.retrieve(q, top_k=3)
    for r in res:
        print(">>> TEXT:", r["text"])
        print(">>> TABLE STRUCT:", r["table"])      # ← 解压后的表格 JSON
        print(">>> META:", r["metadata"])
    # print(f"\n🔍 Query: {q}")
    # for i, r in enumerate(res, 1):
    #     print(f"\n{i}. [score={r['score']}]")
    #     print(r["text"][:300], "...")
    #     print("metadata:", r["metadata"])
    # qvec_table = Embedder().embed_query_table(q)
    # hits = MilvusVectorStore().search_table(qvec_table, top_k=5)

    # for h in hits:
    #     ent = h.entity
    #     table_blob = ent.get("table_blob") or ""
    #     table_json = table = json.loads(zlib.decompress(base64.b64decode(table_blob)).decode())
    #     print("\nTABLE：", table_json)
    #     print("\nTEXT:", ent.get("text")[:300], "...")
    #     print("METADATA:", ent.get("metadata"))
    #     print("SCORE:", h.score)

