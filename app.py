import gradio as gr
import requests
import os

# API地址（运行FastAPI后）
API_URL = os.getenv("API_URL", "http://localhost:8000")

def ask_question(question, use_hyde):
    """调用API回答问题"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "use_hyde": use_hyde},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            sources = "\n\n📚 **参考来源**：\n" + "\n---\n".join([s["content"] for s in data["sources"]])
            return answer, sources
        else:
            return f"错误: {response.status_code}", ""
    except Exception as e:
        return f"请求失败: {str(e)}", ""

def upload_file(file):
    """上传文档"""
    if file is None:
        return "请选择文件"
    
    try:
        with open(file, 'rb') as f:
            files = {'file': (os.path.basename(file), f, 'application/pdf')}
            response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            return f"✅ {response.json()['message']}"
        else:
            return f"上传失败: {response.status_code}"
    except Exception as e:
        return f"上传失败: {str(e)}"

# 创建界面
with gr.Blocks(title="AI智能客服系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI智能客服系统
    ### 基于RAG（检索增强生成）的企业级智能问答助手
    
    **技术栈**：LangChain + Chroma + HyDE + FastAPI + Gradio
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="请输入您的问题", lines=2)
            with gr.Row():
                hyde_checkbox = gr.Checkbox(label="启用HyDE检索增强", value=True)
                submit_btn = gr.Button("发送", variant="primary")
            answer_output = gr.Textbox(label="回答", lines=8)
            sources_output = gr.Textbox(label="参考依据", lines=5)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📄 知识库管理")
            file_upload = gr.File(label="上传PDF/TXT文档", file_types=[".pdf", ".txt"])
            upload_status = gr.Textbox(label="上传状态", interactive=False)
            file_upload.change(upload_file, inputs=file_upload, outputs=upload_status)
            
            gr.Markdown("""
            ### 💡 使用说明
            1. 上传您的文档（PDF或TXT）
            2. 输入问题
            3. 开启HyDE可提升检索准确率
            """)
    
    submit_btn.click(
        ask_question,
        inputs=[question_input, hyde_checkbox],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)