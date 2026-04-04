import logging
from typing import Dict, Any, List, Optional

# 核心检索逻辑 - 使用显式路径防止导入错误
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# 本地模块
from src.vector_store import VectorStore
from src.config import settings

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, documents: Optional[List] = None):
        self.vector_store = VectorStore()
        self.llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0)
        self.final_retriever = None
        if documents:
            self.init_retriever(documents)

    def init_retriever(self, all_documents: List):
        try:
            vector_retriever = self.vector_store.get_retriever()
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 5
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            
            # --- 修复点：去掉 model_name 参数 ---
            # 新版本 langchain-community 的 FlashrankRerank 不再允许直接传 model_name
            # 它会自动使用默认的高效轻量模型（如 ms-marco-tiny-fl-l6）
            compressor = FlashrankRerank() 
            
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )
            logger.info("✅ 检索引擎初始化成功")
        except Exception as e:
            logger.error(f"检索器构建失败: {e}")

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.final_retriever:
            return {"answer": "请先上传 PDF 文档构建知识库。", "sources": []}

        # 构造提示词模板
        prompt = ChatPromptTemplate.from_template("""你是一个专业的工业售后专家。请仅根据[上下文]回答问题。
        
        [上下文]
        {context}
        
        [问题]
        {input}""")

        # 检索出的文档
        retrieved_docs = self.final_retriever.invoke(question)

        # 构造执行链
        chain = (
            {
                "context": lambda x: "\n\n".join([d.page_content for d in retrieved_docs]), 
                "input": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)
        
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in retrieved_docs]
        }