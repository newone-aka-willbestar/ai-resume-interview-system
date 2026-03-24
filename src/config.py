import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API配置（支持多模型）
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
    
    # 模型选择：'openai' / 'zhipu' / 'local'
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    
    # 模型参数
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    
    # RAG参数
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K = int(os.getenv("TOP_K", "3"))  # 检索返回的文档数
    USE_HYDE = os.getenv("USE_HYDE", "true").lower() == "true"  # HyDE增强
    
    # 向量库路径
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")