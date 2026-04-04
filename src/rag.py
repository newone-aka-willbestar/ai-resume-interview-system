import logging
import os
import pickle
from typing import Dict, Any, List, Optional

# 核心检索逻辑
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
        # 建议增加 timeout，防止 Ollama 响应慢导致 500
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL, 
            base_url=settings.OLLAMA_BASE_URL, 
            temperature=0,
            timeout=60  # 设置 60 秒超时
        )
        self.final_retriever = None
        
        # 定义文档缓存路径（存放在向量数据库目录下）
        self.cache_path = os.path.join(settings.VECTORSTORE_PATH, "docs_cache.pkl")

        # --- 核心改进：热启动逻辑 ---
        if documents:
            # 如果是刚上传的文档，直接初始化
            self.init_retriever(documents)
        else:
            # 如果启动时没有传入文档，尝试从本地缓存恢复
            self._try_load_cache()

    def _try_load_cache(self):
        """尝试从本地加载已有的文档缓存以重建检索器"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached_docs = pickle.load(f)
                logger.info(f"📂 发现本地缓存文档 ({len(cached_docs)}个)，正在自动重建检索器...")
                self.init_retriever(cached_docs, save_cache=False)
            except Exception as e:
                logger.error(f"⚠️ 自动热启动失败: {e}")

    def init_retriever(self, all_documents: List, save_cache: bool = True):
        """初始化双路召回 + 精排检索器"""
        try:
            # 1. 向量检索器
            vector_retriever = self.vector_store.get_retriever()
            
            # 2. BM25 检索器（处理关键词）
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            bm25_retriever.k = 5
            
            # 3. 合并双路召回
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            
            # 4. 精排层 (Flashrank)
            compressor = FlashrankRerank() 
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )

            # 5. 保存文档到本地缓存，供下次重启使用
            if save_cache:
                if not os.path.exists(settings.VECTORSTORE_PATH):
                    os.makedirs(settings.VECTORSTORE_PATH)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(all_documents, f)
                logger.info(f"✅ 文档已持久化缓存至: {self.cache_path}")

            logger.info("🚀 检索引擎（双路召回+精排）已就绪")
        except Exception as e:
            logger.error(f"❌ 检索器构建失败: {e}")

    def ask(self, question: str) -> Dict[str, Any]:
        """问答接口：增加防御性逻辑防止 500 报错"""
        if not self.final_retriever:
            return {"answer": "知识库未初始化，请先上传 PDF 手册。", "sources": []}

        logger.info(f"🔍 处理提问: {question}")
        
        try:
            # 1. 检索相关文档
            retrieved_docs = self.final_retriever.invoke(question)

            # 2. 防御性判断：如果没有搜到任何内容，直接回复，不调 LLM
            if not retrieved_docs:
                return {
                    "answer": "抱歉，在目前上传的技术手册中没有找到相关信息。请尝试换一种说法或联系人工客服。",
                    "sources": []
                }

            # 3. 构造上下文
            context = "\n\n".join([d.page_content for d in retrieved_docs])
            
            # 4. 构建提示词
            prompt = ChatPromptTemplate.from_template("""你是一个专业的工业售后专家。请仅根据[参考上下文]回答问题。
            如果上下文中没有提到相关信息，请回答不知道，不要编造答案。
            
            [参考上下文]
            {context}
            
            [问题]
            {input}""")

            # 5. 执行 LLM 链
            chain = (
                {"context": lambda x: context, "input": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)
            
            return {
                "answer": answer,
                "sources": [doc.metadata for doc in retrieved_docs]
            }

        except Exception as e:
            logger.error(f"❌ 问答链路异常: {e}")
            # 面试点：捕获异常并返回友好提示，避免 API 报 500 错误
            return {
                "answer": f"抱歉，系统生成回答时遇到一点小问题（错误详情：{str(e)[:50]}...）。请检查 Ollama 是否开启。",
                "sources": []
            }