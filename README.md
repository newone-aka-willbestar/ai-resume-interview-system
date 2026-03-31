# AI Resume & Interview System

**一个完全本地化的 RAG 智能问答系统**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **完全本地部署 · 数据不出域 · 专为制造业售后场景优化**  
> 支持上传产品手册 PDF，实现基于 RAG（Retrieval-Augmented Generation）的实时智能问答。  
> 仓库名称已更新为 `ai-resume-interview-system`，便于展示 AI 简历分析与面试相关技术能力。

## ✨ 项目亮点

- **100% 本地化部署**：所有模型、向量库、数据均运行在本地服务器，完美满足企业数据安全与隐私合规要求。
- **PDF 智能处理流程**：
  - 自动上传产品手册 PDF
  - 智能分块（RecursiveCharacterTextSplitter）
  - 中文优化 Embedding 向量化
  - 实时向量检索 + 生成式回答
- **中文语义深度优化**：采用 `BAAI/bge-small-zh-v1.5` Embedding 模型，显著提升中文理解准确率。
- **企业级 Prompt 工程**：针对制造业售后服务场景精心设计 Prompt，确保回答专业、准确、友好。
- **前后端分离架构**：后端 FastAPI + 前端 Streamlit，提供流畅的用户体验。
- **大陆网络友好**：内置 hf-mirror 加速 + 手动预下载模型，解决 Hugging Face 下载慢的问题。
- **生产级工程实践**：Docker 支持、环境隔离、错误处理、兼容性优化（LangChain 版本锁定）。

## 🛠 技术栈
| 类别          | 技术/工具                                      | 说明                          |
|---------------|------------------------------------------------|-------------------------------|
| **后端**      | FastAPI + Uvicorn                              | 高性能 API 服务               |
| **前端**      | Streamlit                                      | 快速构建交互式界面            |
| **大模型**    | Ollama + Qwen2-7B                              | 本地运行中文大模型            |
| **向量数据库**| ChromaDB                                       | 本地持久化向量存储            |
| **Embedding** | BAAI/bge-small-zh-v1.5                         | 中文语义向量化                |
| **RAG 框架**  | LangChain (langchain-classic)                  | 确保版本兼容性                |
| **文档处理**  | PyPDFLoader + RecursiveCharacterTextSplitter   | PDF 解析与智能分块            |
| **部署**      | Docker + docker-compose                        | 一键容器化部署                |

## 🚀 快速开始
### 环境要求
- Python 3.8+
- Ollama 已安装并运行（推荐使用 Qwen2-7B）
- 支持 Docker（强烈推荐）
### 1. 克隆仓库
git clone https://github.com/newone-aka-willbestar/ai-resume-interview-system.git
cd ai-resume-interview-system

### 2. 启动 Ollama 服务
ollama serve
ollama pull qwen2:7b

### 3. 安装依赖
pip install -r requirements.txt

### 4. 启动服务
#### 方式一：本地运行（推荐开发）
#启动后端 API
uvicorn src.api:app --reload --port 8000
#新开终端启动前端
streamlit run app.py
#### 方式二：Docker 一键部署（生产推荐）
docker-compose up -d

## 📖 使用说明
- 1.打开前端界面，点击 上传产品手册 按钮，选择 PDF 文件。
- 2.系统自动完成分块、向量化、存入 ChromaDB。
- 3.在聊天框输入售后相关问题（如“产品故障代码 E01 如何处理？”）。
- 4.系统基于 RAG 检索文档并结合 Qwen2 大模型给出专业回答。

## 📁 项目结构
.
├── app.py                    # Streamlit 前端入口
├── src/
│   └── api.py                # FastAPI 后端接口
├── chroma_db/                # ChromaDB 向量数据库（持久化）
├── models/bge-small-zh-v1.5/ # 本地预下载 Embedding 模型
├── .env                      # 环境变量
├── Dockerfile                # Docker 构建文件
├── docker-compose.yml        # Docker Compose 配置
├── requirements.txt          # Python 依赖
└── README.md                 # 本文档

## 🎥 项目演示

### 系统界面截图

![前端主界面 - 上传 PDF](assets/screenshot-main.png)

![api演示](assets/screenshot-chat.png)

### 完整运行演示视频

<video src="assets/demo.mp4" width="100%" controls autoplay loop muted></video>


## 🔧 常见问题与解决方案
- Hugging Face 下载慢 → 使用 hf-mirror 镜像 + 手动预下载模型
- Ollama 502 错误 → 代码中已强制禁用代理
- Embedding 维度冲突 → 删除 chroma_db 目录后重新上传文档
- LangChain 兼容性 → 已锁定 langchain-classic 版本

## 🎯 未来规划
- 向量库管理后台（CRUD + 版本控制）
- Redis 缓存高频问答
- 流式输出（StreamingResponse）
- LangSmith / LangFuse 全链路追踪
- JWT 认证 + 多租户支持
