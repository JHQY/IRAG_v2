import http.client
import json
import time
import traceback
from typing import Any, List, Dict

from fastapi import FastAPI
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
    top_k: int = 3
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


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Main QA endpoint for the frontend.

    1) 基于最近若干轮用户问题 + 当前问题进行 RAG 检索；
    2) 用 prompt_template 生成当前轮 Prompt；
    3) 将历史对话 + 当前 Prompt 一起发送给 LLM 生成答案。
    """
    # 1) 构建用于检索的 query（最近若干轮用户问题 + 当前问题）
    history_user_questions = [
        m.content for m in (req.history or []) if m.role == "user"
    ]
    recent_user_questions = history_user_questions[-3:]  # 只取最近 3 轮用户问题
    rag_query_parts = recent_user_questions + [req.question]
    rag_query = "\n".join(rag_query_parts)

    # 2) RAG 检索
    results = rag.retrieve(rag_query, top_k=req.top_k)

    # 3) 构建参考文本列表
    ref_texts = [r["text"] + "\n" for r in results]

    # 4) 构建当前轮 Prompt（包含当前问题 + 参考内容）
    prompt = auto_build_prompt(req.question, ref_texts, mode=req.mode)

    # 5) 组合多轮对话 messages：history 在前，当前轮在最后
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in (req.history or [])
        if m.role in {"user", "assistant", "system"}
    ]
    messages = history_messages + [{"role": "user", "content": prompt}]

    # 6) 调用 LLM（带简单缓存：仅在无 history 时缓存同一问题的回答）
    cache_key = None
    if not req.history:
        cache_key = json.dumps(
            {"q": req.question, "mode": req.mode, "top_k": req.top_k},
            ensure_ascii=False,
            sort_keys=True,
        )
    if cache_key is not None and cache_key in LLM_CACHE:
        answer_text = LLM_CACHE[cache_key]
    else:
        answer_text = http_client.draw_sample(prompt=messages)
        if cache_key is not None:
            LLM_CACHE[cache_key] = answer_text

    # 7) 返回统一结构
    converted_refs = [RefChunk(**r) for r in results]
    return AskResponse(answer=answer_text, refs=converted_refs)
