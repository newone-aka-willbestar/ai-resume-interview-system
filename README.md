# AI智能客服系统（RAG + HyDE + Ollama）

一个**企业级零成本本地智能客服** Demo，支持 PDF/TXT 上传、HyDE 检索增强、多轮对话、来源追溯。

## ✨ 技术亮点
- **RAG 完整链路**：文档加载 → 智能分块 → Chroma 向量库 + MMR 检索
- **HyDE 检索增强**：生成假设文档后混合检索（准确率显著提升）
- **零 API 费用**：Ollama + Qwen2:7b 本地大模型
- **前后端分离**：FastAPI 后端 + Gradio/Streamlit 前端（已支持切换）
- **Docker 一键部署**

## 🚀 快速启动

```bash
# 1. 克隆项目
git clone https://github.com/newone-aka-willbestar/ai-customer-service.git
cd ai-customer-service

# 2. 安装依赖
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# 3. 下载本地模型
ollama pull qwen2:7b

# 4. 启动后端（终端1）
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 5. 启动前端（终端2）
python app.py