# ingestion/cleaner.py
import pandas as pd
import re

class TableCleaner:
    """
    负责清洗 pdfplumber 解析出来的表格（header + rows）
    处理能力包括：
        - None / 空值修复
        - 多行文本合并
        - 列数不齐整统一补齐
        - header 缺失自动修复
        - 合并单元格的残留问题清理
        - 换行符 / 多空格清理
        - 如果表格无法变成合法 DataFrame，则降级为 text
    """

    def clean_cell(self, x):
        """处理单个单元格"""
        if x is None:
            return ""

        # 统一成字符串
        x = str(x)

        # 去除换行、两侧空白
        x = re.sub(r"\s+", " ", x).strip()

        return x

    def clean_header(self, header):
        """修复 header：None → 空, 长文本截断, 统一清洗规则"""
        if header is None:
            return []

        cleaned = []
        for h in header:
            h = self.clean_cell(h)
            # 若 header 是空，则自动生成 "col_0"
            cleaned.append(h if h else None)

        # 如果 header 全都是 None，则全部改成 col_n
        if all(h is None for h in cleaned):
            cleaned = [f"col_{i}" for i in range(len(cleaned))]
        else:
            # 把 None 补成 col_n
            for i, h in enumerate(cleaned):
                if h is None:
                    cleaned[i] = f"col_{i}"

        return cleaned

    def unify_rows(self, rows, num_cols):
        """确保所有 row 都和 header 列数一致，不足补空，多的合并到最后一格"""
        unified_rows = []
        for r in rows:
            r = [self.clean_cell(x) for x in r]

            if len(r) < num_cols:
                r = r + [""] * (num_cols - len(r))
            elif len(r) > num_cols:
                # 多出的全部塞进最后一个单元格
                r = r[:num_cols-1] + [" ".join(r[num_cols-1:])]

            unified_rows.append(r)

        return unified_rows

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def clean_table(self, header, rows):
        """
        返回:
            df           ← pandas.DataFrame（若失败则 None）
            text_version ← str，用于 fallback 文本检索
        """

        # ------------ 1) 清理 header ------------
        header = [self.clean_cell(h) for h in header]

        # header 全部为空 → 自动补 h_i
        if all(h == "" for h in header):
            header = [f"col_{i}" for i in range(len(header))]
        else:
            for i, h in enumerate(header):
                if h == "":
                    header[i] = f"col_{i}"

        num_cols = len(header)

        # ------------ 2) 清理 rows ------------
        cleaned_rows = []
        for row in rows:
            row = [self.clean_cell(c) for c in row]
            # 对齐列数
            if len(row) < num_cols:
                row += [""] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:num_cols]
            cleaned_rows.append(row)

        # ------------ 3) 横向 Fill-right（合并单元格横向）------------
        for r in range(len(cleaned_rows)):
            for c in range(1, num_cols):
                if cleaned_rows[r][c] == "":
                    cleaned_rows[r][c] = cleaned_rows[r][c - 1]

        # ------------ 4) 纵向 Fill-down（合并单元格向下）------------
        for r in range(1, len(cleaned_rows)):
            for c in range(num_cols):
                if cleaned_rows[r][c] == "":
                    cleaned_rows[r][c] = cleaned_rows[r - 1][c]

        # ------------ 5) 构造 DataFrame ------------
        try:
            df = pd.DataFrame(cleaned_rows, columns=header)
        except Exception:
            df = None

        # ------------ 6) 构造 text_version (fallback) ------------
        lines = []
        lines.append(" | ".join(header))
        for r in cleaned_rows:
            lines.append(" | ".join(r))
        text_version = "\n".join(lines)

        return df, text_version