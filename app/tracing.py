import os

os.environ.setdefault("LANGSMITH_PROJECT", "darglobal-wasalt-chatbot")

from langsmith import traceable


def tracing_enabled():
    return (
        os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def document_load_inputs(inputs):
    return {"data_dir": str(inputs["data_dir"])}


def document_load_outputs(documents):
    return {
        "document_count": len(documents),
        "files": sorted({doc["source_file"] for doc in documents}),
    }


def chat_inputs(inputs):
    request = inputs["req"]
    return {
        "question_length": len(request.message),
        "history_messages": len(request.history),
    }


def chat_outputs(result):
    return {
        "model": result.get("model"),
        "source_count": len(result.get("sources", [])),
        "answer_length": len(result.get("answer", "")),
    }


def context_inputs(inputs):
    return {
        "source_count": len(inputs["documents"]),
        "source_titles": [document["title"] for document in inputs["documents"]],
    }


def context_outputs(context):
    return {"context_characters": len(context)}


def model_request_inputs(inputs):
    messages = inputs["messages"]
    return {
        "model": inputs["model"],
        "message_count": len(messages),
        "prompt_characters": sum(len(message.get("content", "")) for message in messages),
    }


def model_request_outputs(result):
    output = {
        "model": result["model"],
        "status_code": result["status_code"],
        "response_characters": len(result.get("response", "")),
    }
    if "error_code" in result:
        output["error_code"] = result["error_code"]
    if "error_message" in result:
        output["error_message"] = result["error_message"][:200]
    return output


__all__ = ["traceable", "tracing_enabled"]
