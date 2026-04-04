# src/document_loader.py
# 企业级文档加载器 - 专为制造业PDF标准/手册优化（生产可用）
from langchain_community.document_loaders import PyMuPDFLoader  # 推荐替换PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from typing import List
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """企业级文档加载与分割模块"""

    def __init__(self):
        # 企业级推荐参数（可通过settings动态调整）
        self.chunk_size = getattr(settings, "CHUNK_SIZE", 1200)
        self.chunk_overlap = getattr(settings, "CHUNK_OVERLAP", 250)

        # 中文国家标准/手册专用分隔符（关键优化点）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",           # 段落
                "\n",             # 换行
                "。", "！", "？", "；",  # 中文句子结束符
                "5\.", "6\.", "7\.", "8\.", "9\.",  # 保留章节编号 5.1 5.1.1 等
                " ",              # 空格
                "",
            ],
            keep_separator=True,
            add_start_index=True,
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        """加载PDF + 智能分割 + 企业级元数据增强"""
        file_path = Path(file_path)
        logger.info(f"📄 正在加载文档: {file_path.name}")

        # 使用 PyMuPDFLoader（比PyPDFLoader更准，保留更多结构信息）
        loader = PyMuPDFLoader(str(file_path))
        raw_docs = loader.load()

        # 分割
        split_docs = self.text_splitter.split_documents(raw_docs)

        # 企业级元数据增强（生产环境必备，便于后续检索、日志、调试）
        for i, doc in enumerate(split_docs):
            content = doc.page_content.strip()
            # 尝试提取章节标题（适用于GB/T标准格式）
            section_title = content.split("\n")[0][:60] if "\n" in content else content[:60]

            doc.metadata.update({
                "source": file_path.name,
                "page": doc.metadata.get("page", 0) + 1,
                "section_title": section_title,
                "chunk_id": i,
                "total_chunks": len(split_docs),
                "file_type": "pdf",
                "chunk_size": self.chunk_size,
            })

        logger.info(f"✅ 分割完成：{len(raw_docs)} 页 → {len(split_docs)} 个块 "
                    f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return split_docs