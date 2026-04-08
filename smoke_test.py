import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Tuple

# 设置中文字体（解决matplotlib中文显示问题）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class RAGEvaluator:
    """RAG效果评估器，覆盖检索、生成全流程核心维度"""

    def __init__(self, json_path: str):
        """初始化：加载测试数据"""
        self.json_path = json_path
        self.test_data = self._load_json()
        self.summary = self.test_data["summary"]
        self.test_cases = self.test_data["tests"]
        self.evaluation_dimensions = [
            "retrieval_relevance",  # 检索相关性
            "faithfulness",  # 回答忠实度（是否基于检索结果）
            "rejection_quality",  # 拒绝质量（无法回答时的合理性）
            "completeness",  # 回答完整性
            "total"  # 总分
        ]
        # 评估结果存储
        self.eval_results = []

    def _load_json(self) -> Dict:
        """加载JSON测试数据"""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"加载JSON失败: {e}")

    def _check_retrieval_relevance(self, test_case: Dict) -> int:
        """
        评估检索相关性（0-3分）：
        0分：无任何相关检索结果
        1分：1-2条相关但非核心结果
        2分：3-4条相关结果
        3分：≥1条核心相关结果
        """
        prompt = test_case["prompt"]
        retrieval_results = test_case["retrieval"]
        # 提取问题核心关键词（示例：基于简单分词，可替换为专业分词工具）
        prompt_keywords = self._extract_keywords(prompt)
        relevant_count = 0
        core_relevant = False

        for res in retrieval_results:
            res_text = res["text_preview"].lower()
            # 匹配关键词（大小写不敏感）
            match_count = sum([1 for kw in prompt_keywords if kw.lower() in res_text])
            if match_count > 0:
                relevant_count += 1
                # 核心相关：匹配关键词数≥2 或 直接包含问题核心答案
                if match_count >= 2 or self._has_core_answer(res_text, prompt):
                    core_relevant = True

        if core_relevant:
            return 3
        elif relevant_count >= 3:
            return 2
        elif relevant_count >= 1:
            return 1
        else:
            return 0

    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本核心关键词（简单规则，可替换为jieba/TF-IDF）"""
        # 过滤无意义停用词
        stop_words = ["的", "是", "在", "如果", "有", "哪些", "多少", "如何", "吗", "？", "，", "。"]
        # 保留名词/核心动词
        keywords = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]{2,}", text)
        return [kw for kw in keywords if kw not in stop_words]

    def _has_core_answer(self, res_text: str, prompt: str) -> bool:
        """判断检索结果是否包含问题核心答案"""
        # 示例规则（可根据业务场景扩展）
        core_answer_patterns = {
            "等待期": r"90天|缓接期|等待期|观察期",
            "身故理赔": r"理赔|申请|身故保险金|索偿",
            "自杀免责": r"自杀|不保|免责|两年|2年",
            "犹豫期": r"犹豫期|冷静期|反悔|10天|15天|21天",
            "既往症免责": r"既往症|投保前已存在|exclusions|pre-existing",
            "合同变更": r"单方面|变更|修改|投保人权利|条款"
        }
        for core_key, pattern in core_answer_patterns.items():
            if core_key in prompt and re.search(pattern, res_text):
                return True
        return False

    def _check_faithfulness(self, test_case: Dict) -> int:
        """
        评估回答忠实度（0-3分）：
        0分：回答与检索结果完全不符
        1分：部分内容不符
        2分：基本符合但有少量无关信息
        3分：完全基于检索结果，无虚构
        """
        retrieval_text = " ".join([r["text_preview"] for r in test_case["retrieval"]])
        final_answer = test_case["final_answer"]
        model_cot = test_case["model_cot"]

        # 拒绝回答的情况
        if "暂无法确认该问题的答案" in final_answer:
            # 检索无相关内容 → 忠实度3分；检索有相关内容 → 0分
            return 3 if self._check_retrieval_relevance(test_case) == 0 else 0
        # 有回答的情况：检查是否基于检索结果
        answer_keywords = self._extract_keywords(final_answer)
        retrieval_keywords = self._extract_keywords(retrieval_text)
        match_ratio = len(set(answer_keywords) & set(retrieval_keywords)) / len(
            answer_keywords) if answer_keywords else 0

        if match_ratio == 1.0:
            return 3
        elif match_ratio >= 0.7:
            return 2
        elif match_ratio >= 0.3:
            return 1
        else:
            return 0

    def _check_rejection_quality(self, test_case: Dict) -> int:
        """
        评估拒绝质量（0-3分，仅当无法回答时生效）：
        0分：不应拒绝却拒绝 / 拒绝理由不合理
        1分：拒绝理由模糊
        2分：拒绝理由基本合理但不完整
        3分：拒绝理由清晰、合理且建议可行
        """
        final_answer = test_case["final_answer"]
        if "暂无法确认该问题的答案" not in final_answer:
            return 0  # 非拒绝回答，得0分

        retrieval_relevance = self._check_retrieval_relevance(test_case)
        # 检索无相关内容 → 拒绝合理；检索有相关内容 → 拒绝不合理
        if retrieval_relevance > 0:
            return 0

        # 检查拒绝理由完整性
        has_reason = "解释说明" in final_answer and "检索结果与所询问的具体内容不符" in final_answer
        has_suggestion = "查阅对应的保险合同原文" in final_answer or "咨询专业人员" in final_answer

        if has_reason and has_suggestion:
            return 3
        elif has_reason:
            return 2
        else:
            return 1

    def _check_completeness(self, test_case: Dict) -> int:
        """
        评估回答完整性（0-3分）：
        0分：未回答核心问题
        1分：回答部分核心问题
        2分：回答核心问题但缺少细节
        3分：完整回答核心问题且有细节支撑
        """
        prompt = test_case["prompt"]
        final_answer = test_case["final_answer"]
        evaluation = test_case["evaluation"]

        # 拒绝回答的情况
        if "暂无法确认该问题的答案" in final_answer:
            return 0

        # 检查是否覆盖核心问题
        core_keywords = self._extract_keywords(prompt)
        answer_keywords = self._extract_keywords(final_answer)
        core_coverage = len(set(core_keywords) & set(answer_keywords)) / len(core_keywords) if core_keywords else 0

        # 检查是否有细节支撑（证据/解释）
        has_evidence = "[证据]" in final_answer and final_answer.split("[证据]")[1].strip() != "无"
        has_explanation = "[解释说明]" in final_answer and final_answer.split("[解释说明]")[1].strip() != ""

        if core_coverage == 1.0 and (has_evidence or has_explanation):
            return 3
        elif core_coverage >= 0.7:
            return 2
        elif core_coverage >= 0.3:
            return 1
        else:
            return 0

    def evaluate_single_case(self, test_case: Dict) -> Dict:
        """评估单个测试用例"""
        case_id = test_case["id"]
        category = test_case["category"]
        prompt = test_case["prompt"]
        expected = test_case["expected"]
        verdict = test_case["evaluation"]["verdict"]

        # 计算各维度得分
        retrieval_relevance = self._check_retrieval_relevance(test_case)
        faithfulness = self._check_faithfulness(test_case)
        rejection_quality = self._check_rejection_quality(test_case)
        completeness = self._check_completeness(test_case)
        total = retrieval_relevance + faithfulness + rejection_quality + completeness

        # 生成评估结果
        case_result = {
            "case_id": case_id,
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "retrieval_relevance": retrieval_relevance,
            "faithfulness": faithfulness,
            "rejection_quality": rejection_quality,
            "completeness": completeness,
            "total": total,
            "verdict": verdict,
            "manual_reason": test_case["evaluation"]["reason"],
            "auto_reason": self._generate_auto_reason({
                "retrieval_relevance": retrieval_relevance,
                "faithfulness": faithfulness,
                "rejection_quality": rejection_quality,
                "completeness": completeness
            })
        }
        return case_result

    def _generate_auto_reason(self, scores: Dict) -> str:
        """生成自动化评估理由"""
        reasons = []
        if scores["retrieval_relevance"] == 0:
            reasons.append("检索结果无相关内容")
        elif scores["retrieval_relevance"] < 3:
            reasons.append(f"检索相关性不足（得分{scores['retrieval_relevance']}）")

        if scores["faithfulness"] < 3:
            reasons.append(f"回答未完全忠实于检索结果（得分{scores['faithfulness']}）")

        if scores["rejection_quality"] < 3 and scores["rejection_quality"] > 0:
            reasons.append(f"拒绝理由不完整（得分{scores['rejection_quality']}）")

        if scores["completeness"] < 3:
            reasons.append(f"回答完整性不足（得分{scores['completeness']}）")

        return "; ".join(reasons) if reasons else "各维度评估均达标"

    def run_full_evaluation(self) -> None:
        """执行全量评估"""
        print("开始执行RAG效果评估...")
        for case in self.test_cases:
            case_result = self.evaluate_single_case(case)
            self.eval_results.append(case_result)
            print(f"完成用例{case_result['case_id']}评估 | 总分：{case_result['total']}/12")
        print("全量评估完成！")

    def generate_report(self, report_path: str = "rag_evaluation_report.md") -> None:
        """生成可视化评估报告（Markdown + 图表）"""
        # 1. 转换为DataFrame便于分析
        df = pd.DataFrame(self.eval_results)

        # 2. 生成Markdown报告
        report_content = f"""
# RAG效果评估报告
## 1. 整体统计
- 测试执行时间：{self.summary['run_at']}
- 总测试用例数：{self.summary['total_tests']}
- 通过数：{self.summary['pass']}
- 部分通过数：{self.summary['partial']}
- 失败数：{self.summary['fail']}
- 错误数：{self.summary['error']}
- 平均得分：{df['total'].mean():.2f}/12

## 2. 维度得分统计
| 评估维度 | 平均分 | 最高分 | 最低分 |
|----------|--------|--------|--------|
| 检索相关性 | {df['retrieval_relevance'].mean():.2f}/3 | {df['retrieval_relevance'].max()} | {df['retrieval_relevance'].min()} |
| 回答忠实度 | {df['faithfulness'].mean():.2f}/3 | {df['faithfulness'].max()} | {df['faithfulness'].min()} |
| 拒绝质量 | {df['rejection_quality'].mean():.2f}/3 | {df['rejection_quality'].max()} | {df['rejection_quality'].min()} |
| 回答完整性 | {df['completeness'].mean():.2f}/3 | {df['completeness'].max()} | {df['completeness'].min()} |

## 3. 用例详情
| 用例ID | 分类 | 问题 | 自动化总分 | 人工Verdict | 自动化评估理由 |
|--------|------|------|------------|-------------|----------------|
"""
        # 拼接用例详情
        for _, row in df.iterrows():
            report_content += f"\n| {row['case_id']} | {row['category']} | {row['prompt'][:50]}... | {row['total']} | {row['verdict']} | {row['auto_reason']} |"

        # 保存Markdown报告
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"评估报告已保存至：{report_path}")

        # 3. 生成可视化图表
        self._plot_evaluation_results(df)

    def _plot_evaluation_results(self, df: pd.DataFrame) -> None:
        """生成评估结果可视化图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 图1：各维度平均分对比
        dimensions = ["检索相关性", "回答忠实度", "拒绝质量", "回答完整性"]
        avg_scores = [
            df["retrieval_relevance"].mean(),
            df["faithfulness"].mean(),
            df["rejection_quality"].mean(),
            df["completeness"].mean()
        ]
        ax1.bar(dimensions, avg_scores, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
        ax1.set_title("各评估维度平均分", fontsize=14)
        ax1.set_ylim(0, 3)
        ax1.grid(axis="y", alpha=0.3)
        # 标注数值
        for i, score in enumerate(avg_scores):
            ax1.text(i, score + 0.05, f"{score:.2f}", ha="center", fontsize=12)

        # 图2：用例总分分布
        ax2.hist(df["total"], bins=12, color="#9b59b6", edgecolor="black", alpha=0.7)
        ax2.set_title("用例总分分布（满分12）", fontsize=14)
        ax2.set_xlabel("总分")
        ax2.set_ylabel("用例数")
        ax2.grid(axis="y", alpha=0.3)

        # 图3：不同分类用例的平均分
        category_avg = df.groupby("category")["total"].mean()
        ax3.barh(category_avg.index, category_avg.values, color="#1abc9c")
        ax3.set_title("不同分类用例平均分", fontsize=14)
        ax3.set_xlabel("平均分")
        ax3.grid(axis="x", alpha=0.3)
        # 标注数值
        for i, score in enumerate(category_avg.values):
            ax3.text(score + 0.1, i, f"{score:.2f}", va="center", fontsize=12)

        # 图4：人工Verdict分布
        verdict_counts = df["verdict"].value_counts()
        ax4.pie(verdict_counts.values, labels=verdict_counts.index, autopct="%1.1f%%",
                colors=["#27ae60", "#e67e22", "#c0392b"])
        ax4.set_title("人工Verdict分布", fontsize=14)

        plt.tight_layout()
        plt.savefig("rag_evaluation_charts.png", dpi=300, bbox_inches="tight")
        print("可视化图表已保存至：rag_evaluation_charts.png")


if __name__ == "__main__":
    # 初始化评估器
    evaluator = RAGEvaluator(json_path="smoke_test_10.json")

    # 执行全量评估
    evaluator.run_full_evaluation()

    # 生成评估报告（Markdown + 可视化图表）
    evaluator.generate_report()