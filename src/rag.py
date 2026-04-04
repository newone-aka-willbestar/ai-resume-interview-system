import logging
import os
import pickle
from typing import Dict, Any, List, Optional

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.vector_store import VectorStore
from src.config import settings

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, documents: Optional[List] = None):
        self.vector_store = VectorStore()
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL, 
            base_url=settings.OLLAMA_BASE_URL, 
            temperature=0,
            timeout=60
        )
        self.final_retriever = None
        self.cache_path = os.path.join(settings.VECTORSTORE_PATH, "docs_cache.pkl")

        if documents:
            self.init_retriever(documents)
        else:
            self._try_load_cache()

    def _try_load_cache(self):
        """系统启动时，自动从本地硬盘‘复活’检索器"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached_docs = pickle.load(f)
                logger.info(f"📂 发现本地持久化缓存 ({len(cached_docs)}个文本块)，正在复活检索引擎...")
                self.init_retriever(cached_docs, save_cache=False)
            except Exception as e:
                logger.error(f"⚠️ 自动复活失败: {e}")

    def init_retriever(self, all_documents: List, save_cache: bool = True):
        """构建/更新双路检索+精排架构"""
        try:
            vector_retriever = self.vector_store.get_retriever()
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 5
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            
            # 这里的 FlashrankRerank 内部会产生 Numpy 类型，我们后续要处理
            compressor = FlashrankRerank() 
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )

            if save_cache:
                if not os.path.exists(settings.VECTORSTORE_PATH):
                    os.makedirs(settings.VECTORSTORE_PATH)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(all_documents, f)
                logger.info(f"💾 文档已序列化至硬盘，下次启动无需重复上传")

            logger.info("🚀 检索引擎已完全就绪")
        except Exception as e:
            logger.error(f"❌ 检索器构建失败: {e}")

    def sanitize_metadata(self, metadata: dict) -> dict:
        """
        核心修复函数：将 metadata 里的 Numpy 类型转为 Python 原生类型
        解决 500 Internal Server Error (TypeError: 'numpy.float32' object is not iterable)
        """
        new_meta = {}
        for k, v in metadata.items():
            # 关键：检查是否是 Numpy 类型
            if hasattr(v, 'item'): 
                new_meta[k] = v.item() # .item() 将 numpy float/int 转为 python float/int
            elif isinstance(v, dict):
                new_meta[k] = self.sanitize_metadata(v)
            else:
                new_meta[k] = v
        return new_meta

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.final_retriever:
            return {"answer": "知识库为空，请先上传 PDF 文档。", "sources": []}

        try:
            # 1. 检索并精排
            retrieved_docs = self.final_retriever.invoke(question)

            if not retrieved_docs:
                return {"answer": "抱歉，技术手册中未提及相关内容。", "sources": []}

            # 2. 构造上下文
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            
            prompt = ChatPromptTemplate.from_template("""你是一个专业的工业售后专家。请仅根据[参考信息]回答问题。
            如果信息中没有，请直接说不知道，禁止胡乱猜测。
            
            [参考信息]
            {context}
            
            [用户问题]
            {input}""")

            chain = (
                {"context": lambda x: context, "input": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)
            
            # 3. 核心修复：处理返回的元数据，防止 JSON 序列化崩溃
            sanitized_sources = []
            for doc in retrieved_docs:
                meta = self.sanitize_metadata(doc.metadata)
                # 截取一小段内容展示在前端，增加透明度
                meta["content_excerpt"] = doc.page_content[:100] + "..."
                sanitized_sources.append(meta)

            return {
                "answer": answer,
                "sources": sanitized_sources
            }

        except Exception as e:
            logger.error(f"❌ 问答链路异常: {e}")
            return {"answer": f"服务繁忙: {str(e)}", "sources": []}