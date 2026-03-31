# # storage/milvus_store.py
# from pymilvus import (
#     connections, FieldSchema, CollectionSchema,
#     DataType, Collection, utility
# )
# from config.settings import settings
# import numpy as np

# class Chunk:
#     """一个文本或表格块"""
#     def __init__(self, text, metadata):
#         self.text = text
#         self.metadata = metadata


# class MilvusVectorStore:
#     """
#     Milvus 向量存储与检索类
#     - 自动连接 Milvus
#     - 自动创建 collection
#     - 提供 add / search 功能
#     """

#     def __init__(self):
#         self.collection_name = settings.MILVUS_COLLECTION
#         self.dim = settings.MILVUS_DIM

#         # 连接 Milvus
#         connections.connect(
#             alias="default",
#             host=settings.MILVUS_HOST,
#             port=str(settings.MILVUS_PORT)
#         )

#         # 检查 collection 是否存在
#         if not utility.has_collection(self.collection_name):
#             self._create_collection()

#         # 加载 collection
#         self.collection = Collection(self.collection_name)
#         self.collection.load()

#     # ------------------------------------------------------
#     # 创建 collection
#     # ------------------------------------------------------
#     def _create_collection(self):
#         print(f"[Milvus] Creating collection: {self.collection_name}")

#         fields = [
#             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#             FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
#             FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#             FieldSchema(name="metadata", dtype=DataType.JSON)
#         ]

#         schema = CollectionSchema(
#             fields=fields,
#             description="Insurance Knowledge Base"
#         )

#         collection = Collection(name=self.collection_name, schema=schema)

#         # 创建索引
#         index_params = {
#             "index_type": settings.MILVUS_INDEX_TYPE,
#             "metric_type": settings.MILVUS_METRIC_TYPE,
#             "params": {"M": 8, "efConstruction": 64}
#         }

#         collection.create_index(field_name="vector", index_params=index_params)
#         print(f"[Milvus] Collection `{self.collection_name}` created with index.")
#         return collection

#     # ------------------------------------------------------
#     # 插入数据
#     # ------------------------------------------------------
#     def add(self, embeddings, chunks):
#         """
#         向 Milvus 插入一批数据
#         参数：
#           embeddings: List[np.ndarray]  向量
#           chunks: List[Chunk]           对应文本块
#         """
#         if len(embeddings) == 0:
#             return

#         texts = [c.text for c in chunks]
#         metas = [c.metadata for c in chunks]

#         # 插入顺序必须与 collection 定义匹配
#         insert_data = [
#             #[None] * len(embeddings),  # auto_id 主键
#             embeddings,
#             texts,
#             metas
#         ]

#         self.collection.insert(insert_data)
#         self.collection.flush()
#         print(f"[Milvus] ✅ Inserted {len(embeddings)} records.")

#     # ------------------------------------------------------
#     # 向量检索
#     # ------------------------------------------------------
#     def similarity_search(self, query_embedding, top_k=5, filters=None):
#         """
#         执行相似度搜索
#         参数：
#           query_embedding: np.ndarray
#           top_k: 检索结果数
#           filters: dict，可按 metadata 过滤
#         返回：
#           [(Chunk, distance), ...]
#         """
#         search_params = {
#             "metric_type": settings.MILVUS_METRIC_TYPE,
#             "params": {"ef": 50}
#         }

#         expr = None
#         if filters:
#             expr = " and ".join([
#                 f'metadata["{k}"] == "{v}"' for k, v in filters.items()
#             ])

#         results = self.collection.search(
#             data=[query_embedding],
#             anns_field="vector",
#             param=search_params,
#             limit=top_k,
#             expr=expr,
#             output_fields=["text", "metadata"]
#         )

#         hits = []
#         for hit in results[0]:
#             text = hit.entity.get("text")
#             meta = hit.entity.get("metadata")
#             hits.append((Chunk(text, meta), float(hit.distance)))

#         return hits

# from pymilvus import (
#     connections, FieldSchema, CollectionSchema,
#     DataType, Collection, utility
# )
# from config.settings import settings
# import numpy as np
#
# class MilvusVectorStore:
#     """
#     多模态向量存储
#     支持：
#     - 文本向量 bge-m3
#     - 表格向量 TAPAS
#     """
#
#
#     def __init__(self):
#         self.collection_name = "IRAG_MM"
#         self.text_dim = 1024
#         self.table_dim = 768
#
#         connections.connect(
#             alias="default",
#             host=settings.MILVUS_HOST,
#             port=settings.MILVUS_PORT,
#         )
#
#         if not utility.has_collection(self.collection_name):
#             self._create_collection()
#
#         self.collection = Collection(self.collection_name)
#         self.collection.load()
#
#
#     # ------------------------------------------------------------------
#     # 创建全新 IRAG_MM collection
#     # ------------------------------------------------------------------
#     def _create_collection(self):
#
#         print(f"[Milvus] Creating multi-modal collection: {self.collection_name}")
#
#         fields = [
#             FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#
#             FieldSchema(
#                 name="text_vector",
#                 dtype=DataType.FLOAT_VECTOR,
#                 dim=self.text_dim,
#                 description="Text embedding (BGE-M3)"
#             ),
#
#             FieldSchema(
#                 name="table_vector",
#                 dtype=DataType.FLOAT_VECTOR,
#                 dim=self.table_dim,
#                 description="Table embedding (TAPAS)"
#             ),
#
#             FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#             #FieldSchema(name="table_json", dtype=DataType.JSON),
#             FieldSchema(name="table_blob",dtype=DataType.VARCHAR,max_length=65535), # to store table as string
#             FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32),
#             FieldSchema(name="metadata", dtype=DataType.JSON)
#         ]
#
#         schema = CollectionSchema(
#             fields=fields,
#             description="IRAG Multi-Modal Knowledge Base"
#         )
#
#         collection = Collection(self.collection_name, schema)
#
#         # 为两个向量字段分别创建索引
#         index_params = {
#             "index_type": "HNSW",
#             "metric_type": "COSINE",
#             "params": {"M": 8, "efConstruction": 64}
#         }
#
#         collection.create_index("text_vector", index_params)
#         collection.create_index("table_vector", index_params)
#
#         print("[Milvus] Multi-vector collection created.")
#
#     def add_records(self, records):
#         """
#         接收结构化 records，然后写入 Milvus（行模式）。
#         每条记录是一行。
#         """
#
#         import numpy as np
#
#         if not records:
#             return
#
#         text_dim = self.text_dim
#         table_dim = self.table_dim
#
#         # --------------------------------------------------
#         # 强力向量清洗器（递归 flatten + 强制 float + 定长）
#         # --------------------------------------------------
#         def sanitize_vec(v, dim):
#             if v is None:
#                 return [0.0] * dim
#
#             flat = []
#
#             def _flatten(x):
#                 if isinstance(x, (list, tuple, np.ndarray)):
#                     for e in x:
#                         _flatten(e)
#                 else:
#                     try:
#                         flat.append(float(x))
#                     except:
#                         flat.append(0.0)
#
#             _flatten(v)
#
#             if not flat:
#                 flat = [0.0] * dim
#
#             arr = np.array(flat, dtype="float32").reshape(-1)
#
#             if arr.shape[0] < dim:
#                 pad = np.zeros(dim - arr.shape[0], dtype="float32")
#                 arr = np.concatenate([arr, pad])
#             elif arr.shape[0] > dim:
#                 arr = arr[:dim]
#
#             return arr.tolist()
#
#         # --------------------------------------------------
#         # 构造 row-based 插入格式
#         # --------------------------------------------------
#         rows = []
#         for r in records:
#             tv = sanitize_vec(r.get("text_vec"), text_dim)
#             ttv = sanitize_vec(r.get("table_vec"), table_dim)
#
#             row = {
#                 "text_vector": tv,
#                 "table_vector": ttv,
#                 "text": r.get("text") or "",
#                 #"table_json": r.get("table_json") or {},
#                 "table_blob": r.get("table_blob") or "",
#                 "modality": r.get("modality") or "",
#                 "metadata": r.get("metadata") or {},
#             }
#             rows.append(row)
#
#         # --------------------------------------------------
#         # Debug：打印一条 sample 看看结构是否正确
#         # --------------------------------------------------
#         if rows:
#             print("\n[DEBUG] Example Insert Row:")
#             for k, v in rows[0].items():
#                 if isinstance(v, list):
#                     print(f"  {k}: list[{len(v)}]")
#                 else:
#                     print(f"  {k}: {v}")
#
#         # --------------------------------------------------
#         # 最终插入——行模式
#         # --------------------------------------------------
#         self.collection.insert(rows)
#         self.collection.flush()
#
#     # ------------------------------------------------------------------
#     # 搜索（默认 text_vector）
#     # ------------------------------------------------------------------
#     def search_text(self, query_vector, top_k=5):
#
#         results = self.collection.search(
#             data=[query_vector],
#             anns_field="text_vector",
#             param={"metric_type": "COSINE"},
#             limit=top_k,
#             output_fields=["text", "table_blob","modality", "metadata"]
#         )
#
#         return results[0]
#
#
#     # ------------------------------------------------------------------
#     # 搜索表格
#     # ------------------------------------------------------------------
#     def search_table(self, query_vector, top_k=5):
#
#         results = self.collection.search(
#             data=[query_vector],
#             anns_field="table_vector",
#             param={"metric_type": "COSINE"},
#             limit=top_k,
#             output_fields=["text",  "table_blob","modality", "metadata"]
#         )
#
#         return results[0]
# storage/milvus_store.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import zlib, base64, json
from config.settings import settings

class MilvusVectorStore:
    """
    多模态向量存储
    - 支持文本向量 text_vector (BGE-M3)
    - 支持表格向量 table_vector (TAPAS)
    - 可扩展行/列向量
    - 自动压缩表格 JSON
    """

    def __init__(self):
        self.collection_name = "IRAG_MM"
        self.text_dim = 1024
        self.table_dim = 768
        self.row_dim = None      # 可选
        self.column_dim = None   # 可选

        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )

        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)
        self.collection.load()

    # ----------------------
    # 创建 collection
    # ----------------------
    def _create_collection(self):
        print(f"[Milvus] Creating multi-modal collection: {self.collection_name}")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=self.text_dim, description="Text embedding"),
            FieldSchema(name="table_vector", dtype=DataType.FLOAT_VECTOR, dim=self.table_dim, description="Table embedding"),
            # 可选扩展
            # FieldSchema(name="row_vector", dtype=DataType.FLOAT_VECTOR, dim=self.row_dim),
            # FieldSchema(name="column_vector", dtype=DataType.FLOAT_VECTOR, dim=self.column_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="table_blob", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields=fields, description="IRAG Multi-Modal Knowledge Base")
        collection = Collection(self.collection_name, schema)

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64}
        }

        collection.create_index("text_vector", index_params)
        collection.create_index("table_vector", index_params)
        print("[Milvus] Multi-vector collection created.")

    # ----------------------
    # 插入数据
    # ----------------------
    def add_records(self, records):
        """
        接收结构化 records 批量写入 Milvus
        每条记录 dict 格式：
        {
            "text_vec": [...],
            "table_vec": [...],
            "text": str,
            "table_json": dict,  # 会自动压缩为 blob
            "modality": str,
            "metadata": dict
        }
        """

        if not records:
            return

        def sanitize_vec(v, dim):
            if v is None:
                return [0.0] * dim
            flat = []
            def _flatten(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    for e in x: _flatten(e)
                else:
                    try: flat.append(float(x))
                    except: flat.append(0.0)
            _flatten(v)
            arr = np.array(flat, dtype="float32").reshape(-1)
            if arr.shape[0] < dim:
                arr = np.concatenate([arr, np.zeros(dim - arr.shape[0], dtype="float32")])
            elif arr.shape[0] > dim:
                arr = arr[:dim]
            return arr.tolist()

        rows = []
        for r in records:
            tv = sanitize_vec(r.get("text_vec"), self.text_dim)
            ttv = sanitize_vec(r.get("table_vec"), self.table_dim)
            table_blob = r.get("table_blob") or ""
            if not table_blob and r.get("table_json"):
                # 压缩表格 JSON
                jstr = json.dumps(r["table_json"])
                compressed = zlib.compress(jstr.encode("utf-8"))
                table_blob = base64.b64encode(compressed).decode("utf-8")

            row = {
                "text_vector": tv,
                "table_vector": ttv,
                "text": r.get("text") or "",
                "table_blob": table_blob,
                "modality": r.get("modality") or "",
                "metadata": r.get("metadata") or {},
            }
            rows.append(row)

        self.collection.insert(rows)
        self.collection.flush()
        print(f"[Milvus] ✅ Inserted {len(rows)} records.")

    # ----------------------
    # 解压表格 blob
    # ----------------------
    @staticmethod
    def decompress_table_blob(blob: str) -> dict:
        if not blob:
            return {}
        data = base64.b64decode(blob)
        raw = zlib.decompress(data)
        return json.loads(raw.decode("utf-8"))

    # ----------------------
    # 统一搜索接口
    # ----------------------
    def search(self, query_vector, top_k=5, modality="text"):
        if modality == "table":
            anns_field = "table_vector"
            expr = 'modality == "table"'
        elif modality in {"text", "row", "column"}:
            anns_field = "text_vector"
            expr = f'modality == "{modality}"'
        else:
            anns_field = "text_vector"
            expr = None
        results = self.collection.search(
            data=[query_vector],
            anns_field=anns_field,
            param={"metric_type": "COSINE"},
            limit=top_k,
            expr=expr,
            output_fields=["text","table_blob","modality","metadata"]
        )
        return results[0]
