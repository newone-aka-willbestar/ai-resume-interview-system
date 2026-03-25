import streamlit as st
import requests
import time
from typing import Dict

st.set_page_config(page_title="华科制造智能售后客服", page_icon="robot", layout="wide")

API_BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key-2026"   # ← 在 .env 中配置 settings.API_KEY

with st.sidebar:
    st.title("系统设置")
    api_url = st.text_input("后端 API 地址", value=API_BASE_URL, disabled=True)
    api_key_input = st.text_input("API Key", value=API_KEY, type="password")
    
    st.divider()
    st.caption("文档上传")
    uploaded_files = st.file_uploader("上传产品手册 / PDF", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("上传到知识库", type="primary"):
        with st.spinner("正在上传并处理..."):
            success_count = 0
            for file in uploaded_files:
                files = {"file": (file.name, file.getvalue(), file.type)}
                headers = {"X-API-Key": api_key_input}
                try:
                    resp = requests.post(f"{api_url}/upload", files=files, headers=headers)
                    if resp.status_code == 200:
                        success_count += 1
                        st.success(f"{file.name} 上传成功")
                    else:
                        st.error(f"{file.name} 上传失败: {resp.json().get('detail')}")
                except Exception as e:
                    st.error(f"上传失败: {e}")
            if success_count == len(uploaded_files):
                st.success(f"全部 {success_count} 个文件上传完成！")

    if st.button("清空聊天记录"):
        st.session_state.messages = []
        st.rerun()

st.title("华科制造智能售后客服系统")
st.markdown("基于 RAG 的本地智能客服 · 支持文档上传")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("参考来源"):
                for src in message["sources"]:
                    st.write(f"**{src.get('metadata', {}).get('source', '未知文档')}**")

if prompt := st.chat_input("请输入您的问题，例如：这个设备的保修期是多久？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在思考中..."):
            try:
                headers = {"X-API-Key": api_key_input}
                payload = {"question": prompt}
                response = requests.post(f"{API_BASE_URL}/ask", json=payload, headers=headers, timeout=300)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "抱歉，我暂时无法回答。")
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("参考来源"):
                            for i, src in enumerate(sources, 1):
                                st.write(f"**来源{i}**：{src.get('metadata', {}).get('source', '未知文档')}")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                else:
                    st.error(f"请求失败: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"发生错误: {e}")

st.caption("提示：第一次提问前请先上传 PDF 文档")