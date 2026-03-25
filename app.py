import streamlit as st
import requests
import time
from typing import Dict

# ==================== 配置 ====================
st.set_page_config(
    page_title="华科制造智能售后客服",
    page_icon="🤖",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key-2026"

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("⚙️ 系统设置")
    api_url = st.text_input("后端 API 地址", value=API_BASE_URL, disabled=True)
    api_key_input = st.text_input("API Key", value=API_KEY, type="password")
    
    st.divider()
    st.caption("📄 文档上传")
    uploaded_files = st.file_uploader(
        "上传产品手册 / PDF（支持多个）",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 上传到知识库", type="primary"):
            with st.spinner("正在上传并处理文档..."):
                success_count = 0
                for file in uploaded_files:
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    headers = {"X-API-Key": api_key_input}
                    try:
                        resp = requests.post(f"{api_url}/upload", files=files, headers=headers)
                        if resp.status_code == 200:
                            success_count += 1
                            st.success(f"✅ {file.name} 上传成功")
                        else:
                            st.error(f"❌ {file.name} 上传失败: {resp.json().get('detail', resp.text)}")
                    except Exception as e:
                        st.error(f"❌ 上传失败: {e}")
                if success_count == len(uploaded_files):
                    st.success(f"🎉 全部 {success_count} 个文件上传完成！")

    # 清空聊天按钮
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()

# ==================== 主界面 ====================
st.title("🤖 华科制造智能售后客服系统")
st.markdown("基于 RAG 的本地智能客服 · 支持文档上传 · 实时问答")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 参考来源"):
                for src in message["sources"]:
                    st.write(f"**{src.get('source', '未知文档')}**")

# 输入框
if prompt := st.chat_input("请输入您的问题，例如：这个设备的保修期是多久？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 正在思考中（CPU模式可能稍慢）..."):
            try:
                headers = {"X-API-Key": api_key_input}
                payload = {
                    "question": prompt,
                    "use_hyde": True
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{api_url}/ask",
                    json=payload,
                    headers=headers,
                    timeout=300          # ← 改成 5 分钟，解决 10054 错误
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "抱歉，我暂时无法回答。")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 参考来源"):
                            for i, src in enumerate(sources, 1):
                                st.write(f"**来源{i}**：{src.get('source', '未知文档')}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    st.caption(f"⏱️ 耗时 {time.time()-start_time:.2f} 秒")
                else:
                    error_msg = response.json().get("detail", response.text)
                    st.error(f"❌ 请求失败: {error_msg}")
                    
            except requests.exceptions.Timeout:
                st.error("❌ 回答超时（模型生成太慢）。建议换用 qwen2:3b 小模型")
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接后端服务！请确认 uvicorn 和 Ollama 都在运行")
            except Exception as e:
                # 专门处理 WinError 10054
                if "10054" in str(e):
                    st.error("❌ Ollama 回答太慢被强制断开连接。\n\n**解决办法**：\n1. 换用 qwen2:3b 模型（推荐）\n2. 或在 .env 里把 OLLAMA_MODEL 改小")
                else:
                    st.error(f"❌ 发生错误: {e}")

# 底部提示
st.caption("💡 提示：第一次提问前请先在侧边栏上传 PDF 文档")