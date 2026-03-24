from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.vector_store import VectorStoreManager
from src.hyde import HyDE
from src.config import Config
from typing import Dict, Any

class RAGChain:
    """完整的RAG问答链（Ollama + HyDE完全生效版）"""

    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.retriever = self.vector_manager.get_retriever()
        self.llm = ChatOllama(
            model=Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE if hasattr(Config, 'TEMPERATURE') else 0.3
        )
        self.hyde = HyDE() if Config.USE_HYDE else None

        # Prompt模板（保留你原来的专业客服风格）
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的智能客服助手。请严格根据以下参考内容回答用户问题。

【参考内容】
{context}

【回答要求】
1. 如果参考内容中有答案，请基于参考内容回答
2. 如果参考内容中没有答案，请说"抱歉，当前知识库中没有找到相关信息"
3. 回答要简洁、准确、友好
4. 必要时可以分点列出

【用户问题】{question}
【回答】""")
        ])

    def _format_docs(self, docs):
        """格式化检索到的文档"""
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def _retrieve_with_hyde(self, question: str):
        """HyDE完全生效版：先生成假设文档 → 用假设文档+原问题进行混合检索"""
        if not self.hyde or not Config.USE_HYDE:
            return self.retriever.invoke(question)

        # 1. 生成假设文档
        hypo_doc = self.hyde.generate_hypothetical_document(question)

        # 2. 用假设文档检索（增强召回）
        hypo_results = self.retriever.invoke(hypo_doc)

        # 3. 用原问题再检索（保证相关性）
        original_results = self.retriever.invoke(question)

        # 4. 合并去重（MMR已在上层retriever处理，这里简单合并）
        combined = hypo_results + original_results
        # 去重（按内容）
        seen = set()
        unique_docs = []
        for doc in combined:
            content_key = doc.page_content[:100]
            if content_key not in seen:
                seen.add(content_key)
                unique_docs.append(doc)

        return unique_docs[:Config.TOP_K * 2]   # 取更多再让MMR过滤

    def get_chain(self):
        """构建LCEL链（推荐方式）"""
        def retrieve_with_context(question: str):
            docs = self._retrieve_with_hyde(question)
            context = self._format_docs(docs)
            return {"question": question, "context": context}

        chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: retrieve_with_context(x["question"]) if isinstance(x, dict) else retrieve_with_context(x))
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def answer(self, question: str) -> Dict[str, Any]:
        """对外统一回答接口（返回答案 + 来源）"""
        docs = self._retrieve_with_hyde(question)
        context = self._format_docs(docs)

        # 使用链或直接invoke
        response = self.llm.invoke(
            self.prompt.format_messages(question=question, context=context)
        )

        return {
            "answer": response.content,
            "sources": [
                {
                    "content": doc.page_content[:300],
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }