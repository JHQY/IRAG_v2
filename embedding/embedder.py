# '''Text embedding placeholder'''
# import torch
# from sentence_transformers import SentenceTransformer
# from config.settings import settings
# import numpy as np

# class Embedder:
#     def __init__(self):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=device)

#     def embed_text(self, texts):
#         return np.array(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False),dtype=np.float32)
    
#     def embed_query(self, query):
#         return np.array(self.model.encode([query], convert_to_numpy=True, show_progress_bar=False),dtype=np.float32)[0]()  # Load environment variables

"""
Embedder for IRAG multi-modal pipeline
- 文本 → BGE / SentenceTransformer embedding
- 表格 → TAPAS embedding (structure-aware)
"""

import torch
from transformers import AutoTokenizer, AutoModel
from transformers import TapasTokenizer, TapasModel
import numpy as np
import pandas as pd


class Embedder:
<<<<<<< HEAD
    def __init__(self):
        device = "cpu"
        if torch.cuda.is_available():
            try:
                _ = torch.randn(1, device="cuda")
                device = "cuda"
            except Exception:
                device = "cpu"
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=device)
=======
>>>>>>> 7baeaa1 (new-version-with-table)

    def __init__(self):

        # ----- 文本模型（BGE、m3、sentence-BERT都可以） -----
        self.text_model_name = "BAAI/bge-m3"
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_model.eval()

        # ----- 表格模型（TAPAS 论文级表格 embedding） -----
        self.table_model_name = "google/tapas-base"
        self.table_tokenizer = TapasTokenizer.from_pretrained(self.table_model_name)
        self.table_model = TapasModel.from_pretrained(self.table_model_name)
        self.table_model.eval()

        # 是否使用 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model.to(self.device)
        self.table_model.to(self.device)



    # ----------------------------------------------------------------------
    # 文本 embedding
    # ----------------------------------------------------------------------
    def embed_text(self, texts):
<<<<<<< HEAD
        return np.array(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False),dtype=np.float32)
    
    def embed_query(self, query):
        return np.array(self.model.encode([query], convert_to_numpy=True, show_progress_bar=False),dtype=np.float32)[0]  # Load environment variables
=======
        """
        输入: texts = [str, str, ...]
        输出: np.ndarray (N, dim)
        """

        inputs = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]   # CLS embedding

        return embeddings.cpu().numpy()



    # ----------------------------------------------------------------------
    # 表格 embedding（论文级 TAPAS）
    # ----------------------------------------------------------------------
    def embed_table(self, headers, rows):
        """
        输入:
            headers = ["Benefit", "Coverage", ...]
            rows = [
                ["Major Illness", "100%", "0"],
                ["Minor Illness", "30%", "200"]
            ]

        输出: np.ndarray (dim=768)
        """

        

        # --- 1. 构造 DataFrame（TAPAS 需要） ---
        df = pd.DataFrame(rows, columns=headers)

        # TAPAS 必须有 queries 参数
        inputs = self.table_tokenizer(
            table=df,
            queries=["table embedding query"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.table_model(**inputs)
            embedding = outputs.pooler_output  # shape (1, 768)

        return embedding.cpu().numpy()[0]

    def embed_query_table(self, query: str):
        """
        将 query 转成 TAPAS 所需的 DataFrame 格式
        """

        # 用 DataFrame 更安全
        df = pd.DataFrame({"QUERY": [query]})

        inputs = self.table_tokenizer(
            table=df,                     # 👈 NOW DataFrame
            queries=["query"],            # TAPAS 必填
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.table_model(**inputs)
            emb = outputs.pooler_output  # (1, dim)

        return emb.cpu().numpy()[0]

>>>>>>> 7baeaa1 (new-version-with-table)
