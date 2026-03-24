from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.config import Config

class HyDE:
    """HyDE (Hypothetical Document Embeddings) 检索增强"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=0.2,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""请针对以下问题，写一段假设的答案文档。答案不需要完全准确，
但应该包含可能出现在真实文档中的关键词和表述。直接输出答案内容，不要加额外说明。

问题：{question}

假设答案："""
        )
    
    def generate_hypothetical_document(self, question: str) -> str:
        """生成假设文档"""
        chain = self.prompt | self.llm
        response = chain.invoke({"question": question})
        return response.content