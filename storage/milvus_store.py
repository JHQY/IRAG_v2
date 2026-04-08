# storage/milvus_store.py
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    DataType,
    CollectionSchema,
    utility
)
from config.settings import settings


class MilvusVectorStore:
    def __init__(self, collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        # 【关键】统一为768维（和你原来的一致）
        self.vector_dim = 768
        self.text_dim = self.vector_dim
        self.table_dim = self.vector_dim
        # 连接Milvus
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT
        )

        # 创建集合（如果不存在）
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        self.collection = Collection(self.collection_name)
        # 加载集合到内存
        self.collection.load()

    # storage/milvus_store.py
    def _create_collection(self):
        """调大VARCHAR字段长度，彻底解决超限问题"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=20),
            # 【修改】text字段从4096→8192，留足余量
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            # 【修改】table_blob字段从16384→32768，表格压缩后可能较长
            FieldSchema(name="table_blob", dtype=DataType.VARCHAR, max_length=32768),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields, "IRAG_MM 多模态保险知识库")
        collection = Collection(self.collection_name, schema)

        # 创建索引
        collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
        )
        print(f"[Milvus] 集合 {self.collection_name} 创建成功")

    def add_records(self, records: list):
        """【修复】和你原来的indexer.py完全兼容"""
        entities = []
        for record in records:
            entities.append({
                "modality": record["modality"],
                "text": record["text"],
                "table_blob": record["table_blob"],
                "vector": record["vector"],  # 所有模态都用这个字段
                "metadata": record["metadata"],
            })
        self.collection.insert(entities)
        self.collection.flush()

    def search(self, query_vector: list, modality: str = None, top_k: int = 10):
        """
        【修复】单向量字段+modality过滤
        这是你原来的正确搜索逻辑，完全兼容indexer.py
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 64}
        }

        # 如果指定了modality，就过滤
        expr = f"modality == '{modality}'" if modality else None

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",  # 唯一向量字段
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "table_blob", "modality", "metadata"]
        )

        return results[0]