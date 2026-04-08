import numpy as np
from typing import List, Dict, Any
import torch
class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """
        初始化嵌入模型，必须与indexer.py使用的模型完全一致
        建议使用 bge-large-zh-v1.5，保险领域效果最佳
        """
        self.model_name = model_name
        # 这里使用你实际的嵌入模型初始化代码
        # 示例：假设使用sentence-transformers
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)
        
        # 为了演示，这里使用模拟向量，实际请替换为真实模型
        self.text_dim = 768
        self.table_dim = 768
        print(f"[Embedder] 初始化完成，模型: {model_name}")
        
    def embed_text(self, texts: List[str]) -> List[np.ndarray]:
        """
        文本嵌入，与indexer.py完全一致
        用于：文本chunk、表格的row级、表格的column级
        """
        # 实际代码：
        # return self.model.encode(texts, normalize_embeddings=True)
        
        # 模拟代码（请替换为真实模型）
        return [np.random.randn(self.text_dim).astype("float32") for _ in texts]

    def embed_table(self, header: List[str], rows: List[List[str]]) -> np.ndarray:
        """
        表格级嵌入，与indexer.py完全一致
        用于：表格的table级模态
        """
        # 方法1：如果有专门的表格嵌入模型，使用它
        # return self.table_model.encode(header, rows)
        
        # 方法2：如果没有专门模型，将表格拼接为文本后embed_text（与indexer.py一致）
        table_text = " | ".join(header) + "\n"
        for row in rows:
            table_text += " | ".join(row) + "\n"
        
        # 实际代码：
        # return self.model.encode(table_text, normalize_embeddings=True)
        
        # 模拟代码（请替换为真实模型）
        return np.random.randn(self.table_dim).astype("float32")

    def embed_query_table(self, query: str) -> np.ndarray:
        """
        查询时的表格嵌入（可选，用于table级查询）
        """
        return self.embed_text([query])[0]