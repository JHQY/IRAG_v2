"""
Embedder for the IRAG multi-modal pipeline.
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import TapasModel, TapasTokenizer


class Embedder:
    def __init__(self):
        self.text_model_name = "BAAI/bge-m3"
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_model.eval()

        self.table_model_name = "google/tapas-base"
        self.table_tokenizer = TapasTokenizer.from_pretrained(self.table_model_name)
        self.table_model = TapasModel.from_pretrained(self.table_model_name)
        self.table_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_model.to(self.device)
        self.table_model.to(self.device)

        tokenizer_max_length = getattr(self.text_tokenizer, "model_max_length", 512)
        if not isinstance(tokenizer_max_length, int) or tokenizer_max_length <= 0 or tokenizer_max_length > 8192:
            tokenizer_max_length = 8192
        self.text_max_length = tokenizer_max_length

    def embed_text(self, texts):
        """
        Args:
            texts: list[str]
        Returns:
            np.ndarray with shape (N, dim)
        """
        if not texts:
            return np.zeros((0, self.text_model.config.hidden_size), dtype=np.float32)

        normalized_texts = [self._safe_text(text) for text in texts]
        inputs = self.text_tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]

        return embeddings.cpu().numpy().astype(np.float32)

    def embed_table(self, headers, rows):
        """
        Encode a table into a single TAPAS embedding.
        """
        df = self._table_to_dataframe(headers, rows)
        if df.empty:
            return np.zeros((self.table_model.config.hidden_size,), dtype=np.float32)

        inputs = self.table_tokenizer(
            table=df,
            queries=["What does this table describe?"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.table_model(**inputs)
            emb = outputs.pooler_output

        return emb.cpu().numpy()[0].astype(np.float32)

    def embed_query_table(self, query: str):
        """
        Encode a table-oriented query into TAPAS-compatible space.
        """
        safe_query = self._safe_text(query)
        if not safe_query:
            return np.zeros((self.table_model.config.hidden_size,), dtype=np.float32)

        # TAPAS always expects a table alongside the natural-language query.
        dummy_table = pd.DataFrame({"context": ["table retrieval"]})
        inputs = self.table_tokenizer(
            table=dummy_table,
            queries=[safe_query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.table_model(**inputs)
            emb = outputs.pooler_output

        return emb.cpu().numpy()[0].astype(np.float32)

    @staticmethod
    def _safe_text(value) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _table_to_dataframe(self, headers, rows) -> pd.DataFrame:
        safe_headers = [self._safe_text(h) or f"column_{idx}" for idx, h in enumerate(headers or [])]
        safe_rows = rows or []

        if not safe_headers and safe_rows:
            max_cols = max(len(row or []) for row in safe_rows)
            safe_headers = [f"column_{idx}" for idx in range(max_cols)]

        if not safe_headers:
            return pd.DataFrame()

        normalized_rows = []
        width = len(safe_headers)
        for row in safe_rows:
            row_values = list(row or [])
            row_values = [self._safe_text(v) for v in row_values[:width]]
            if len(row_values) < width:
                row_values.extend([""] * (width - len(row_values)))
            normalized_rows.append(row_values)

        try:
            return pd.DataFrame(normalized_rows, columns=safe_headers)
        except ValueError:
            return pd.DataFrame()
