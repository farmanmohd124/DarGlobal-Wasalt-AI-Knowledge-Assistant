"""Hybrid retrieval: BM25 (sparse) + dense embeddings -> Reciprocal Rank Fusion -> cross-encoder rerank."""
import math, os, re
from collections import Counter
import numpy as np
from .tracing import traceable

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
CACHE = os.getenv("MODEL_CACHE", "/tmp/models")

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())


def stage_inputs(inputs):
    instance = inputs.get("self")
    query = inputs.get("q", "")
    return {
        "query_length": len(query),
        "document_count": len(instance.docs) if instance is not None else None,
        "candidate_count": len(inputs.get("indices", [])),
    }


def score_outputs(scores):
    if scores is None:
        return {"scored_documents": 0, "top_scores": []}
    return {
        "scored_documents": int(len(scores)),
        "top_scores": [round(float(scores[i]), 4) for i in np.argsort(-scores)[:5]],
    }


def ranking_inputs(inputs):
    return {"input_rankings": [len(ranking) for ranking in inputs["rankings"]]}


def rerank_outputs(ranking):
    return {
        "ranked_document_indices": [int(index) for index, _ in ranking[:10]],
        "top_scores": [round(float(score), 4) for _, score in ranking[:10]],
    }


def index_outputs(ranking):
    return {"ranked_document_indices": [int(index) for index in ranking[:10]]}


def search_outputs(result):
    documents, trace = result
    return {
        "selected_documents": [document["title"] for document in documents],
        "retrieval_mode": trace["mode"],
    }


def query_embedding_inputs(inputs):
    query = inputs.get("q", "")
    return {"query_length": len(query)}


def query_embedding_outputs(vector):
    return {"dimensions": int(vector.shape[0])}


def similarity_inputs(inputs):
    return {
        "document_count": inputs["document_count"],
        "embedding_dimensions": int(inputs["query_vector"].shape[0]),
    }


class Retriever:
    def __init__(self, docs, embedder=None, reranker=None, load_models=True):
        self.docs = docs
        for d in docs: d["tokens"] = tok(f'{d["title"]} {d["company"]} {d["text"]}')
        self.N = len(docs); self.avgdl = sum(len(d["tokens"]) for d in docs) / max(self.N, 1)
        self.df = Counter(t for d in docs for t in set(d["tokens"]))
        self.embedder, self.reranker, self.status = embedder, reranker, {}
        if load_models and embedder is None: self.embedder = self._load("embed")
        if load_models and reranker is None:
            if os.getenv("DISABLE_RERANK") == "1":
                self.status["rerank"] = "disabled by env"
            else:
                self.reranker = self._load("rerank")
        self.vecs = None
        if self.embedder is not None:
            try:
                texts = [f'{document["title"]}. {document["text"]}' for document in docs]
                self.vecs = self._embed_documents(texts)
            except Exception as e: self.embedder = None; self.status["embed"] = f"failed: {e}"
        self.status.setdefault("embed", EMBED_MODEL if self.vecs is not None else "disabled")
        self.status.setdefault("rerank", RERANK_MODEL if self.reranker is not None else "disabled")

    @traceable(
        name="Load retrieval model",
        run_type="chain",
        process_inputs=lambda inputs: {"model_kind": inputs["kind"]},
        process_outputs=lambda model: {"loaded": model is not None},
    )
    def _load(self, kind):
        if os.getenv("DISABLE_NEURAL") == "1":
            self.status[kind] = "disabled by env"; return None
        try:
            if kind == "embed":
                from fastembed import TextEmbedding
                return TextEmbedding(EMBED_MODEL, cache_dir=CACHE)
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            return TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=CACHE)
        except Exception as e:
            self.status[kind] = f"unavailable: {type(e).__name__}"; return None

    @traceable(
        name="Embed knowledge documents",
        run_type="embedding",
        process_inputs=lambda inputs: {"document_count": len(inputs["texts"])},
        process_outputs=lambda vectors: {
            "vector_count": int(vectors.shape[0]),
            "dimensions": int(vectors.shape[1]),
        },
    )
    def _embed_documents(self, texts):
        vectors = np.array(list(self.embedder.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    @traceable(name="BM25 search", run_type="retriever", process_inputs=stage_inputs, process_outputs=score_outputs)
    def bm25(self, q):
        qt = tok(q); out = []
        for i, d in enumerate(self.docs):
            tf = Counter(d["tokens"]); dl = len(d["tokens"]); s = 0.0
            for t in qt:
                if t in tf:
                    idf = math.log(1 + (self.N - self.df[t] + .5) / (self.df[t] + .5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (.25 + .75 * dl / self.avgdl))
            out.append(s)
        return np.array(out)

    @traceable(name="Dense search", run_type="retriever", process_inputs=stage_inputs, process_outputs=score_outputs)
    def dense(self, q):
        if self.vecs is None: return None
        query_vector = self._embed_query(q)
        query_vector = query_vector / np.linalg.norm(query_vector)
        return self._dense_similarity(query_vector, len(self.docs))

    @traceable(
        name="Embed retrieval query",
        run_type="embedding",
        process_inputs=query_embedding_inputs,
        process_outputs=query_embedding_outputs,
    )
    def _embed_query(self, query):
        vectors = self.embedder.query_embed([query]) if hasattr(self.embedder, "query_embed") else self.embedder.embed([query])
        return np.array(list(vectors))[0]

    @traceable(
        name="Dense similarity search",
        run_type="retriever",
        process_inputs=similarity_inputs,
        process_outputs=score_outputs,
    )
    def _dense_similarity(self, query_vector, document_count):
        return self.vecs @ query_vector

    @staticmethod
    @traceable(
        name="Reciprocal rank fusion",
        run_type="chain",
        process_inputs=ranking_inputs,
        process_outputs=index_outputs,
    )
    def rrf(rankings, k=60):
        score = Counter()
        for r in rankings:
            for rank, i in enumerate(r): score[i] += 1.0 / (k + rank + 1)
        return [i for i, _ in score.most_common()]

    @traceable(name="Cross-encoder rerank", run_type="chain", process_inputs=stage_inputs, process_outputs=rerank_outputs)
    def _rerank(self, q, indices):
        texts = [f'{self.docs[i]["title"]}. {self.docs[i]["text"]}' for i in indices]
        scores = list(self.reranker.rerank(q, texts))
        return sorted(
            ((index, score) for index, score in zip(indices, scores)),
            key=lambda item: -item[1],
        )

    @traceable(name="Hybrid retrieval", run_type="retriever", process_inputs=stage_inputs, process_outputs=search_outputs)
    def search(self, q, k=4, pool=10):
        trace = {"query": q}
        b = self.bm25(q); b_rank = [int(i) for i in np.argsort(-b)[:pool] if b[i] > 0]
        d = self.dense(q); d_rank = [int(i) for i in np.argsort(-d)[:pool]] if d is not None else []
        fused = self.rrf([r for r in (b_rank, d_rank) if r])[:pool]
        trace["bm25"] = [self.docs[i]["title"] for i in b_rank[:5]]
        trace["dense"] = [self.docs[i]["title"] for i in d_rank[:5]]
        trace["fused"] = [self.docs[i]["title"] for i in fused[:5]]
        final = fused[:k]
        if self.reranker is not None and fused:
            try:
                order = self._rerank(q, fused)
                final = [i for i, _ in order[:k]]
                trace["reranked"] = [{"title": self.docs[i]["title"], "score": round(float(s), 3)} for i, s in order[:k]]
            except Exception as e: trace["rerank_error"] = str(e)
        trace["mode"] = "hybrid+rerank" if "reranked" in trace else ("hybrid" if d_rank else "bm25")
        return [self.docs[i] for i in final], trace
