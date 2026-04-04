import torch
import json
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    """
    智能多模态 Reranker：
    1. 自动识别输入是纯文本还是表格 JSON
    2. 对表格进行结构化解析与格式化（Markdown/Key-Value）
    3. 支持单模型或双模型（文本/表格分别用不同模型）架构
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-base",
            table_model_name: str = None,
            device: str = None,
            table_format: str = "markdown"  # 可选: "markdown", "keyvalue", "json"
    ):
        """
        :param model_name: 通用文本重排模型
        :param table_model_name: (可选) 专门针对表格优化的重排模型
        :param device: 计算设备
        :param table_format: 表格序列化方式
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.table_format = table_format

        # 初始化通用模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 初始化表格专用模型（如果提供）
        self.use_dual_model = table_model_name is not None
        if self.use_dual_model:
            self.table_tokenizer = AutoTokenizer.from_pretrained(table_model_name)
            self.table_model = AutoModelForSequenceClassification.from_pretrained(table_model_name)
            self.table_model.to(self.device)
            self.table_model.eval()

    def _is_json_table(self, text: str) -> dict:
        """
        检测输入字符串是否为 retriever 传来的表格 JSON
        返回解析后的 dict，若不是则返回 None
        """
        if not text or not text.strip().startswith('{'):
            return None
        try:
            data = json.loads(text)
            # 简单的 schema 校验：必须包含 header 和 rows
            if isinstance(data, dict) and "header" in data and "rows" in data:
                return data
        except Exception:
            pass
        return None

    def _format_table(self, table_data: dict) -> str:
        """
        将表格 dict 格式化为对模型友好的字符串
        """
        header = table_data.get("header", [])
        rows = table_data.get("rows", [])

        if not header:
            return json.dumps(table_data, ensure_ascii=False)

        if self.table_format == "markdown":
            # 方案1: Markdown 表格 (对通用语言模型最友好)
            lines = []
            lines.append(f"| {' | '.join(map(str, header))} |")
            lines.append(f"| {' | '.join(['---'] * len(header))} |")
            for row in rows:
                # 确保行长度一致
                padded_row = list(row) + [''] * (len(header) - len(row))
                lines.append(f"| {' | '.join(map(str, padded_row[:len(header)]))} |")
            return "\n".join(lines)

        elif self.table_format == "keyvalue":
            # 方案2: Key-Value 对 (适合列数少但行数多的表，或针对检索特定行)
            lines = []
            for row_idx, row in enumerate(rows):
                cells = [f"{header[i]}: {row[i]}" for i in range(min(len(header), len(row)))]
                lines.append(f"[Row {row_idx + 1}] " + "; ".join(cells))
            return "\n".join(lines)

        # 兜底：原始 JSON
        return json.dumps(table_data, ensure_ascii=False)

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """
        输入:
            query: str
            texts: List[str] (可以是纯文本，也可以是表格 JSON 字符串)
        输出:
            List[float] 对应每个文本的相关性分数
        """
        # 1. 预处理：分离文本和表格，分别构建输入对
        # 即使是单模型，我们也希望把表格变成更好的格式
        processed_texts = []
        is_table_flags = []  # 标记哪些是表格

        for text in texts:
            table_data = self._is_json_table(text)
            if table_data:
                # 是表格，进行格式化
                processed_texts.append(self._format_table(table_data))
                is_table_flags.append(True)
            else:
                # 是普通文本
                processed_texts.append(text)
                is_table_flags.append(False)

        # 2. 推理策略
        if not self.use_dual_model:
            # 策略 A：单模型流 (默认)
            # 所有输入（格式化后的表格+文本）一起进通用模型
            pairs = [[query, t] for t in processed_texts]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                scores = self.model(**inputs).logits.squeeze(-1)

            return scores.cpu().tolist()

        else:
            # 策略 B：双模型流 (进阶)
            # 文本进文本模型，表格进表格模型
            # 注意：这需要分别 batch 处理以保证效率

            # 收集索引
            text_indices = [i for i, is_table in enumerate(is_table_flags) if not is_table]
            table_indices = [i for i, is_table in enumerate(is_table_flags) if is_table]

            all_scores = [0.0] * len(texts)

            # 处理文本
            if text_indices:
                text_batch = [processed_texts[i] for i in text_indices]
                text_pairs = [[query, t] for t in text_batch]
                inputs = self.tokenizer(text_pairs, padding=True, truncation=True, max_length=512,
                                        return_tensors="pt").to(self.device)
                with torch.no_grad():
                    scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()
                for idx, score in zip(text_indices, scores):
                    all_scores[idx] = score

            # 处理表格
            if table_indices:
                table_batch = [processed_texts[i] for i in table_indices]
                table_pairs = [[query, t] for t in table_batch]
                # 表格通常较长，适当放宽 max_length
                inputs = self.table_tokenizer(table_pairs, padding=True, truncation=True, max_length=1024,
                                              return_tensors="pt").to(self.device)
                with torch.no_grad():
                    scores = self.table_model(**inputs).logits.squeeze(-1).cpu().tolist()
                for idx, score in zip(table_indices, scores):
                    all_scores[idx] = score

            return all_scores