from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import VectorStoreManager
from src.hyde import HyDE
from src.config import settings
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGChain:
    """完整的RAG问答链（华科制造智能售后客服场景）"""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.retriever = self.vector_manager.get_retriever()
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE
        )
        self.hyde = HyDE() if settings.USE_HYDE else None

        # === 华科制造企业场景 Prompt（面试亮点）===
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是**华科制造**官方智能售后客服助手。
请严格根据参考内容，用专业、友好、规范的语气回答。
必须做到：
1. 直接引用产品型号、故障代码、保修条款。
2. 回答末尾标注来源（文档名称/页码）。
3. 无法回答时礼貌引导用户提供更多信息或转人工。

【华科制造知识库参考内容】
{context}

【用户问题】{question}
【回答】""")
        ])

    def _format_docs(self, docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def _retrieve_with_hyde(self, question: str):
        if not self.hyde or not settings.USE_HYDE:
            return self.retriever.invoke(question)

        hypo_doc = self.hyde.generate_hypothetical_document(question)
        hypo_results = self.retriever.invoke(hypo_doc)
        original_results = self.retriever.invoke(question)

        combined = hypo_results + original_results
        seen = set()
        unique_docs = []
        for doc in combined:
            content_key = doc.page_content[:100]
            if content_key not in seen:
                seen.add(content_key)
                unique_docs.append(doc)
        return unique_docs[:settings.TOP_K * 2]

    def answer(self, question: str) -> Dict[str, Any]:
        logger.info(f"收到问题: {question}")
        docs = self._retrieve_with_hyde(question)
        context = self._format_docs(docs)

        response = self.llm.invoke(
            self.prompt.format_messages(question=question, context=context)
        )

        logger.info(f"回答完成，引用 {len(docs)} 个片段")
        return {
            "answer": response.content,
            "sources": [
                {"content": doc.page_content[:300] + "...", "metadata": doc.metadata}
                for doc in docs
            ]
        }