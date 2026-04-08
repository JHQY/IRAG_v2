import http.client
import json
import time
import traceback
from typing import Any, List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval.retriever import RAGInterface
from prompt_template import auto_build_prompt


class HttpsApi:
    def __init__(self, host: str, key: str, model: str, timeout: int = 20, **kwargs: Any) -> None:
        """Simple HTTPS API client for OpenAI-compatible chat completions."""
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args: Any, **kwargs: Any) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 4096),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt,
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'IRAG-Frontend/1.0',
                    'Content-Type': 'application/json',
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                return data['choices'][0]['message']['content']
            except Exception:
                print(
                    f'Error when calling LLM API: {traceback.format_exc()}.'
                    f'You may check your API host and API key.'
                )
                time.sleep(2)
                continue


# -----------------------------
# FastAPI app & global deps
# -----------------------------

app = FastAPI(title="IRAG QA API")

# Mount static frontend assets directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 15
    mode: str = "expert"  # expert / customer / academic / json
    history: List[Message] = []


class RefChunk(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    refs: List[RefChunk]


# Initialize RAG + LLM client once at startup
rag = RAGInterface()

# NOTE: these should ideally come from environment variables or config
LLM_HOST = "api.bltcy.top"
LLM_KEY = "sk-Clt5fdhN9xAT9sk2aj6MRCEgF8Zv7ahy3KQP1RK5PqHRGpCP"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
http_client = HttpsApi(host=LLM_HOST, key=LLM_KEY, model=LLM_MODEL)
LLM_CACHE: Dict[str, str] = {}


@app.get("/")
async def index() -> FileResponse:
    """Serve the Vue frontend."""
    return FileResponse("frontend/index.html")


# 新增：处理检索结果的辅助函数（核心修复source为空问题）
def process_retrieval_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理RAG检索结果，确保metadata的source字段有默认值，过滤无效结果
    :param results: 原始检索结果
    :return: 处理后的检索结果
    """
    processed = []
    for res in results:
        # 1. 过滤空文本/空score的无效结果
        if not res.get("text") or not isinstance(res.get("score"), (int, float)):
            continue

        # 2. 确保metadata字段存在且为字典
        metadata = res.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # 3. 给source字段设置默认值（解决source为空问题）
        metadata["source"] = metadata.get("source") or "未知来源"

        # 4. 重构结果项，确保字段完整
        processed_item = {
            "text": res["text"].strip(),
            "score": float(res["score"]),  # 确保score是浮点数
            "metadata": metadata
        }
        processed.append(processed_item)

    return processed


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        # 1. 构建 RAG 查询（保留你的原有逻辑）
        history_user_questions = [m.content for m in (req.history or []) if m.role == "user"]
        recent_user_questions = history_user_questions[-3:]
        rag_query_parts = recent_user_questions + [req.question]
        rag_query = "\n".join(rag_query_parts)

        # 2. RAG 检索（保留）
        results = rag.retrieve(rag_query, top_k=req.top_k)

        # 3. 处理检索结果（保留你刚才加的代码，确保 source 正常）
        processed_results = process_retrieval_results(results)

        # ====================== 【关键新增：过滤出有效文本，构建上下文】 ======================
        # 提取所有非空的 text 内容，组成传给 LLM 的 context
        valid_contexts = []
        for item in processed_results:
            text = item.get("text", "").strip()
            if text:
                valid_contexts.append(text)

        # 如果没有有效上下文，直接返回错误信息
        if not valid_contexts:
            return AskResponse(
                answer="抱歉，未能检索到有效的参考资料。",
                refs=processed_results
            )

        # 4. 【核心修复】构建 Prompt（完全替换你的 auto_build_prompt 逻辑）
        # 我们直接在这里写死 Prompt，强制 LLM 回答，不依赖外部模板
        system_prompt = """
你是一个专业的保险咨询助手。
你的任务是：**严格、准确地从参考资料中提取信息回答用户的问题**。
**绝对不可以说“参考资料中未包含相关信息”，除非参考资料中完全没有提到关键字词。**
回答时，请优先使用具体的数字、比例和条款名称。
"""

        user_prompt = f"""
用户问题：{req.question}

以下是参考资料，请根据这些资料回答问题。
参考资料：
{chr(10).join(valid_contexts)}

请直接回答问题，不要提及参考资料来源。
如果有数字或比例，请明确指出。
"""

        # 5. 组合消息（保留你的 HTTP 客户端逻辑）
        messages = [{"role": "system", "content": system_prompt}]
        if req.history:
            for msg in req.history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_prompt})

        # 6. 调用 LLM（保留你原有的 http_client 代码）
        answer_text = http_client.draw_sample(prompt=messages)

        # 7. 返回
        return AskResponse(answer=answer_text, refs=processed_results)

    except Exception as e:
        traceback.print_exc()
        # 出错时也返回合法格式，避免前端报错
        return AskResponse(answer=f"处理出错：{str(e)}", refs=[])