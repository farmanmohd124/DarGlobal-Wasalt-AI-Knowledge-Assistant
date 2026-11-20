import os
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
c=os.getenv("MODEL_CACHE","/srv/models")
TextEmbedding(os.getenv("EMBED_MODEL","BAAI/bge-small-en-v1.5"),cache_dir=c)
TextCrossEncoder(model_name=os.getenv("RERANK_MODEL","Xenova/ms-marco-MiniLM-L-6-v2"),cache_dir=c)
print("models cached")
