# prompt_template.py
# -*- coding: utf-8 -*-
"""
Unified Chinese-English Prompt Templates for IRAG
"""

import re
from typing import List

# ---------- 中文模板 ----------
def build_prompt_cn(query: str, ref_text: List[str], mode: str = "expert") -> str:
    context = "\n".join(ref_text)

    if mode == "expert":
        prompt = f"""
你是一名资深保险领域专家。请严格根据以下参考资料回答用户的问题：
1. 仅依据参考资料内容回答，不得编造；
2. 若资料中无相关信息，请回答：“参考资料中未包含相关信息”；
3. 回答应详细、条理清晰，可分点说明；
4. 尽量引用参考资料中的具体内容或片段。

【参考资料】
{context}

【问题】
{query}

【回答】
"""
    elif mode == "customer":
        prompt = f"""
你是一名保险客服代表。请根据以下参考资料，用通俗、友好的语气回答用户的问题。
若参考资料中未提及，请回复：“资料中未包含相关信息”。

【参考资料】
{context}

【用户问题】
{query}

【答复】
"""
    elif mode == "academic":
        prompt = f"""
你是一名保险政策研究人员，请依据以下参考资料撰写一份逻辑清晰、书面化的分析。
1. 尽量引用参考资料中的具体内容或片段进行分析；
2. 分点说明各个相关信息；
3. 若资料不足以得出明确结论，请在最后说明：“参考资料不足以支撑明确结论”。

【参考资料】
{context}

【研究问题】
{query}

【学术回答】
"""
    elif mode == "json":
        prompt = f"""
请根据以下参考资料回答问题，并以合法 JSON 格式输出：
{{
  "answer": "详细回答",
  "sources": ["引用片段1", "引用片段2"]
}}

参考资料：
{context}

问题：
{query}
"""
    else:
        prompt = f"""
请基于以下参考内容回答问题：
若资料中未提及答案，请说明“参考资料中未包含相关信息”。

参考内容：
{context}

问题：
{query}

回答：
"""
    return prompt.strip()

# ---------- 英文模板 ----------
def build_prompt_en(query: str, ref_text: List[str], mode: str = "expert") -> str:
    context = "\n".join(ref_text)

    if mode == "expert":
        prompt = f"""
You are a senior insurance expert. Please answer the user's question strictly based on the following reference content:
1. Use only the provided information;
2. If no relevant information is found, reply: "No related information found in the reference materials.";
3. Provide a detailed, well-structured answer, using bullet points or numbered lists where appropriate;
4. Whenever possible, quote specific content from the references.

[Reference Content]
{context}

[Question]
{query}

[Answer]
"""
    elif mode == "customer":
        prompt = f"""
You are an insurance customer service assistant. Based on the following references, answer the customer's question in a friendly, easy-to-understand tone.
If the reference materials do not include relevant information, reply:
"No related information found in the reference materials."

[Reference Materials]
{context}

[Customer Question]
{query}

[Response]
"""
    elif mode == "academic":
        prompt = f"""
You are a researcher specializing in insurance policies. Based on the reference materials, write a formal, logically structured analysis:
1. Quote specific content from the references wherever possible;
2. Explain your analysis point by point;
3. If the materials are insufficient to draw a clear conclusion, mention at the end: "The available materials are insufficient to draw a clear conclusion."


[References]
{context}

[Research Question]
{query}

[Academic Answer]
"""
    elif mode == "json":
        prompt = f"""
Please answer the following question strictly in valid JSON format:
{{
  "answer": "detailed answer here",
  "sources": ["quoted fragment 1", "quoted fragment 2"]
}}

Below are the reference materials:
{context}

Question:
{query}
"""
    else:
        prompt = f"""
Please answer the question based on the following references.
If no relevant information is found, reply:
"No related information found in the reference materials."

References:
{context}

Question:
{query}

Answer:
"""
    return prompt.strip()

# ---------- 自动语言检测 ----------
def auto_build_prompt(query: str, ref_text: List[str], mode: str = "expert") -> str:
    """
    自动检测语言并调用中英文模板。
    中文 → 使用 build_prompt_cn()
    英文 → 使用 build_prompt_en()
    """
    if re.search(r'[\u4e00-\u9fa5]', query):
        return build_prompt_cn(query, ref_text, mode)
    else:
        return build_prompt_en(query, ref_text, mode)