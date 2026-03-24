import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_PROVIDER = "ollama"                    # 新增：强制本地
    CHAT_MODEL = "qwen2:7b"                    # Ollama模型
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K = int(os.getenv("TOP_K", "4"))
    USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"

    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./chroma_db")