import torch
import json
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    """
    保险表格专用Reranker（带调试模式）
    """

    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-large",
            device: str = None,
            table_boost: float = 0.3,  # 表格分数加成
            table_max_length: int = 1024,
            debug: bool = True  # 【新增】调试开关，默认开启，看完日志后可改为False
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.table_boost = table_boost
        self.table_max_length = table_max_length
        self.debug = debug

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _is_json_table(self, text: str) -> dict:
        if not text or len(text) < 20 or not text.strip().startswith('{'):
            return None
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "header" in data and "rows" in data and len(data["rows"]) > 0:
                return data
        except Exception:
            pass
        return None

    def _format_insurance_table(self, table_data: dict, query: str) -> str:
        header = table_data.get("header", [])
        rows = table_data.get("rows", [])
        query_lower = query.lower()

        column_keywords = ["金额", "赔付", "保额", "保费", "最高", "最低", "平均", "总计", "限额"]
        is_column_query = any(k in query_lower for k in column_keywords)

        if is_column_query:
            col_texts = []
            for col_idx, col_name in enumerate(header):
                col_name_lower = col_name.lower()
                prefix = "【关键列】" if any(k in col_name_lower for k in column_keywords) else ""
                col_values = [str(row[col_idx]) for row in rows if len(row) > col_idx]
                col_texts.append(f"{prefix}{col_name}：{', '.join(col_values)}")
            return "\n".join(col_texts)

        row_keywords = ["哪个", "什么", "包含", "包括", "范围"]
        is_row_query = any(k in query_lower for k in row_keywords)

        if is_row_query:
            row_texts = []
            for row_idx, row in enumerate(rows):
                cells = [f"{header[i]}：{row[i]}" for i in range(min(len(header), len(row)))]
                row_texts.append(f"保障项目{row_idx + 1}：{' | '.join(cells)}")
            return "\n".join(row_texts)

        md_lines = [f"| {' | '.join(map(str, header))} |"]
        md_lines.append(f"| {' | '.join(['---'] * len(header))} |")
        for row in rows[:10]:
            md_lines.append(f"| {' | '.join(map(str, row))} |")
        return "\n".join(md_lines)

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []

        # --- 1. 预处理 ---
        processed_texts = []
        is_table_flags = []
        original_texts = []  # 保存原始文本用于调试

        for text in texts:
            original_texts.append(text)
            table_data = self._is_json_table(text)
            if table_data:
                processed = self._format_insurance_table(table_data, query)
                processed_texts.append(processed)
                is_table_flags.append(True)
            else:
                processed_texts.append(text)
                is_table_flags.append(False)

        # --- 2. 模型推理 ---
        pairs = [[query, t] for t in processed_texts]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.table_max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

        # --- 3. 分数加成 ---
        final_scores = []
        for score, is_table in zip(scores, is_table_flags):
            if is_table:
                final_scores.append(score + self.table_boost)
            else:
                final_scores.append(score)

        # --- 4. 调试打印 (核心新增部分) ---
        if self.debug:
            self._print_debug_info(query, original_texts, processed_texts, is_table_flags, scores, final_scores)

        return final_scores

    def _print_debug_info(self, query, original_texts, processed_texts, is_table_flags, raw_scores, boosted_scores):
        """漂亮地打印调试信息"""
        print("\n" + "=" * 80)
        print(f"🔍 [Rerank Debug] 用户查询: {query}")
        print("=" * 80)

        # 打包并排序（按最终分数从高到低）
        candidates = list(zip(range(len(original_texts)), original_texts, processed_texts, is_table_flags, raw_scores,
                              boosted_scores))
        candidates.sort(key=lambda x: x[5], reverse=True)  # 按加成后的分数排序

        for idx, (orig_idx, orig_text, proc_text, is_table, raw_s, boost_s) in enumerate(candidates):
            status = "🟢 表格 (TABLE)" if is_table else "⚪ 文本 (TEXT)"
            print(f"\n--- 排名 {idx + 1} (原始索引: {orig_idx}) | {status} ---")
            print(
                f"   原始模型分: {raw_s:.4f} | 加成后分数: {boost_s:.4f} (加成: +{self.table_boost if is_table else 0})")

            # 打印处理后的文本预览（前200字符）
            preview = proc_text[:200].replace('\n', ' ')
            print(f"   输入模型文本: {preview}{'...' if len(proc_text) > 200 else ''}")

            # 如果是表格，额外打印原始JSON的表头
            if is_table:
                try:
                    data = json.loads(orig_text)
                    print(f"   [表格结构] 表头: {data.get('header', [])} | 行数: {len(data.get('rows', []))}")
                except:
                    pass

        print("\n" + "=" * 80 + "\n")