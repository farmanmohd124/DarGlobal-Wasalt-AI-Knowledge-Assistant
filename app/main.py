import json, os
from pathlib import Path
import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .retrieval import Retriever
from .tracing import (
    chat_inputs,
    chat_outputs,
    context_inputs,
    context_outputs,
    document_load_inputs,
    document_load_outputs,
    model_request_inputs,
    model_request_outputs,
    traceable,
    tracing_enabled,
)

BASE = Path(__file__).parent
DATA = BASE.parent / "data"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "openrouter/free,inclusionai/ling-3.2-flash:free,google/gemma-4-26b-a4b-it:free").split(",")]

@traceable(
    name="Load knowledge documents",
    run_type="chain",
    process_inputs=document_load_inputs,
    process_outputs=document_load_outputs,
)
def load_documents(data_dir):
    documents = []
    for filename in ("knowledge.json", "scraped.json"):
        path = data_dir / filename
        if path.exists():
            file_documents = json.loads(path.read_text(encoding="utf-8"))
            for document in file_documents:
                document["source_file"] = filename
            documents.extend(file_documents)
    return documents


@traceable(
    name="Build answer context",
    run_type="chain",
    process_inputs=context_inputs,
    process_outputs=context_outputs,
)
def build_context(documents):
    return "\n\n".join(
        f"[{document['company']} – {document['title']}] {document['text']}"
        for document in documents
    )


docs = load_documents(DATA)
retriever = Retriever(docs)

SYSTEM = ("You are an assistant for questions about DarGlobal (luxury real estate developer) and Wasalt "
          "(Saudi AI real estate platform). Answer ONLY from the provided context. If the context does not "
          "contain the answer, say you don't have that information and suggest contacting the company. "
          "Never invent prices, dates or figures. Be concise, and mention the source page title when useful.")

app = FastAPI(title="DarGlobal & Wasalt AI Knowledge Assista")

class Chat(BaseModel):
    message: str
    history: list[dict] = []

@app.get("/")
def index(): return FileResponse(BASE / "static" / "index.html")

@app.get("/health")
def health(): return {"status": "ok", "chunks": len(docs), "retrieval": retriever.status, "llm_configured": bool(os.getenv("OPENROUTER_API_KEY")), "models": MODELS, "langsmith_tracing": tracing_enabled()}

@app.post("/api/chat")
@traceable(
    name="Chat request",
    run_type="chain",
    process_inputs=chat_inputs,
    process_outputs=chat_outputs,
)
async def chat(req: Chat):
    hits, trace = retriever.search(req.message)
    sources = [{"title": h["title"], "company": h["company"], "url": h["url"]} for h in hits]
    key = os.getenv("OPENROUTER_API_KEY")
    if not hits:
        return {"answer": "I don't have information on that in my DarGlobal/Wasalt knowledge base.", "sources": [], "model": None, "trace": trace}
    ctx = build_context(hits)
    if key:
        msgs = [{"role": "system", "content": SYSTEM}] + req.history[-6:] + [
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {req.message}"}]
        async with httpx.AsyncClient(timeout=60) as client:
            for m in MODELS:
                try:
                    result = await request_model(client, m, msgs, key)
                    if result["status_code"] == 200 and result.get("answer", "").strip():
                        return {"answer": result["answer"].strip(), "sources": sources, "model": m, "trace": trace}
                except Exception:
                    continue
    # fallback: retrieval-only answer
    return {"answer": "(LLM unavailable – showing best matching facts)\n\n" + hits[0]["text"], "sources": sources, "model": None, "trace": trace}


@traceable(
    name="OpenRouter model request",
    run_type="llm",
    process_inputs=model_request_inputs,
    process_outputs=model_request_outputs,
)
async def request_model(client, model, messages, key):
    response = await client.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": os.getenv("APP_URL", "http://localhost"),
            "X-Title": "DarGlobal-Wasalt Chatbot",
        },
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 500},
    )
    result = {"model": model, "status_code": response.status_code, "response": response.text}
    if response.status_code == 200:
        result["answer"] = response.json()["choices"][0]["message"]["content"]
    else:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            if "code" in error:
                result["error_code"] = error["code"]
            if "message" in error:
                result["error_message"] = str(error["message"])
    return result
