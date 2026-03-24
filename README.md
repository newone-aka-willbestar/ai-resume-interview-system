本项目结合AI打造企业级应用场景
# 华科制造智能售后客服系统（企业级 RAG）

**项目定位**：为智能制造企业打造的**本地零成本AI售后客服系统**，完美解决产品手册分散、客服响应慢、数据安全风险等问题。

### 🎯 业务场景
华科制造（模拟头部工业自动化企业）每年处理上万条售后咨询。传统方式依赖人工检索PDF手册，效率低、易出错。本系统支持上传产品规格书、维修SOP、保修政策等文档，客户/员工通过自然语言提问，系统自动检索+生成专业回答，并附上来源。

**核心价值**：
- 24×7自助服务，预计减少人工工单70%
- 数据完全本地化（Ollama + Chroma），符合制造业数据不出域要求
- 支持 HyDE 检索增强 + 来源追溯 + API 密钥认证

### 技术亮点（已企业化）
- FastAPI + Pydantic Settings + 结构化日志
- Ollama 本地模型（零 API 费用）
- Docker + docker-compose 一键部署
- API Key 认证 + CI/CD + pytest
- 模块化架构，可快速扩展 Redis/PostgreSQL

### 🚀 快速启动（推荐 Docker）

```bash
# 1. 启动服务
docker compose up -d

# 2. 进入 Ollama 容器拉模型（第一次较慢）
docker exec -it ollama ollama pull qwen2:7b

# 3. 访问 Swagger 文档
http://localhost:8000/docs


#本项目流程
1.根据ai给出的企业场景打造适应场景的项目
确定流程与技术栈，并针对技术栈完成requirement和.env
2.结合教程与github上的项目进行初步的搭建（config.py）
llm->Embedding->向量库->api->企业级搭建（api安全与日志）
3.文档分割，针对输入的文档进行文档处理
4.向量库建立与储存结果