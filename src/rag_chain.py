from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from src.vector_store import VectorStoreManager
from src.hyde import HyDE
from src.config import Config
from typing import Dict, Any

class RAGChain:
    """完整的RAG问答链"""
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.retriever = self.vector_manager.get_retriever()
        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.hyde = HyDE() if Config.USE_HYDE else None
        
        # 定义Prompt模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的智能客服助手。请根据以下参考内容回答用户问题。

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
        """使用HyDE增强的检索（简化版：仅用原问题检索）"""
        # 注意：这里为了简化，并未真正使用生成的假设文档去检索。
        # 实际生产中可以：先生成假设文档 -> 用假设文档检索 -> 结合原问题排序。
        # 为了演示HyDE思想，我们只是调用生成函数但不改变检索，你可以自行扩展。
        if self.hyde:
            _ = self.hyde.generate_hypothetical_document(question)
            # 这里你可以将生成的文档作为额外上下文，或用来扩充查询。
        return self.retriever.invoke(question)
    
    def get_chain(self):
        """构建Runnable链（LangChain表达式语言）"""
        def retrieve(question):
            if self.hyde and Config.USE_HYDE:
                return self._retrieve_with_hyde(question)
            return self.retriever.invoke(question)
        
        chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: retrieve(x["question"])) | self._format_docs
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def answer(self, question: str) -> Dict[str, Any]:
        """回答问题，返回答案和来源"""
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)
        
        response = self.llm.invoke(
            self.prompt.format_messages(question=question, context=context)
        )
        
        return {
            "answer": response.content,
            "sources": [{"content": doc.page_content[:200], "metadata": doc.metadata} for doc in docs]
        }