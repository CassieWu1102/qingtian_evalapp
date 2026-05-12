import streamlit as st
import docx
import PyPDF2
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载后端环境变量（从 .env 文件读取，对前端用户不可见）
load_dotenv()

# 从环境变量中获取后端配置
BACKEND_API_KEY = os.getenv("API_KEY", "")
BACKEND_BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
BACKEND_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

st.set_page_config(page_title="招标组织方案评估系统", page_icon="📝", layout="wide")

# --- 1. 文档解析模块 ---
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"文件解析失败: {str(e)}")
    return text

# --- 2. AI 评估逻辑 ---
def evaluate_document(bidding_text, plan_text, api_key, base_url="https://api.openai.com/v1", model="gpt-4o"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 针对科大讯飞“青天”大模型的定制 Prompt
    system_prompt = """
    你是一个资深的招标与评标专家，深谙国内招标法及各大企业（特别是科大讯飞“青天”大模型自动化评标系统）的评标规则和偏好。
    用户的目的是让这份【招标组织方案】在经过“青天”大模型等 AI 或人工评标时获得高分，顺利通过。
    
    现在，用户提供了两份文档：
    1. 【招标文件】：规定了本次招标的要求、背景和评分标准。
    2. 【招标组织方案】：用户根据招标文件编写的响应方案。
    
    请结合【招标文件】的具体要求，对用户的【招标组织方案】进行全面评估，并按以下结构输出你的建议：
    
    1. 🎯 **需求响应度与一致性检查（关键）**
       - 严格对比【招标文件】，指出【招标组织方案】中是否存在遗漏项、未响应项或矛盾点（废标风险）。
       - 检查资质要求、时间节点、核心技术/商务指标是否一一对应且合规。
    
    2. 🤖 **AI 评标系统（“青天”模型）友好度分析**
       - 关键词匹配度：方案中是否充分提取并包含【招标文件】中的核心得分关键词，以及通用加分词（如“公平、公正、公开”、“全流程追溯”、“信息安全”等）。
       - 结构化响应：AI 提取信息的难易程度，是否采用了清晰的“点对点”响应结构，是否存在排版混乱导致 AI 误判或漏判的风险。
    
    3. 💡 **专业修改建议（逐条列出）**
       - 给出具体、可执行的修改方案。
       - 针对偏离招标文件要求或表述不够严谨的地方，提供【修改前】与【修改后】的对比示例。
       
    4. 📈 **最终评分预估与总结**
       - 给出 1-100 分的预估分数（基于对招标文件的响应度以及 AI 评标友好度），并简短总结。
    """
    
    user_content = f"【招标文件】内容如下：\n{bidding_text}\n\n====================\n\n【招标组织方案】内容如下：\n{plan_text}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 评估过程出错: {str(e)}"

# --- 3. Streamlit 前端交互界面 ---
def main():
    st.title("📝 招标组织方案智能评估系统")
    st.markdown("""
    本应用旨在帮助您评估和优化**招标组织方案**。系统将对比您的**招标文件**，从专业角度排查风险，并针对 **科大讯飞“青天”大模型** 等自动化评标系统的偏好提供响应度分析和修改建议，助力方案顺利通关。
    """)
    
    with st.sidebar:
        st.header("⚙️ 系统状态")
        
        # 仅显示后端是否已配置的状态，不暴露具体 Key
        if BACKEND_API_KEY:
            st.success(f"✅ AI 评估引擎已连接\n\n当前模型: `{BACKEND_MODEL_NAME}`")
        else:
            st.error("⚠️ AI 评估引擎未配置，请联系系统管理员在服务器端进行设置。")
            
        st.markdown("---")
        st.markdown("### 📌 支持的文件格式\n- PDF (`.pdf`)\n- Word (`.docx`)\n- 文本 (`.txt`)")

    st.subheader("1. 📤 上传文档")
    col1, col2 = st.columns(2)
    with col1:
        bidding_file = st.file_uploader("1️⃣ 请上传【招标文件】", type=["pdf", "docx", "txt"])
    with col2:
        plan_file = st.file_uploader("2️⃣ 请上传【招标组织方案】", type=["pdf", "docx", "txt"])

    if bidding_file is not None and plan_file is not None:
        st.success("两份文件均已上传成功！")
        
        with st.spinner("正在解析文件内容..."):
            bidding_text = extract_text_from_file(bidding_file)
            plan_text = extract_text_from_file(plan_file)
            
        if bidding_text and plan_text:
            with st.expander("📄 查看提取的文档内容"):
                st.markdown("### 招标文件内容预览")
                st.text_area("招标文件", bidding_text, height=200, key="bidding_preview")
                st.markdown("### 招标组织方案内容预览")
                st.text_area("招标组织方案", plan_text, height=200, key="plan_preview")
                
            st.subheader("2. 🤖 开始评估")
            if st.button("🚀 提交 AI 专家评估", type="primary"):
                if not BACKEND_API_KEY:
                    st.warning("⚠️ 服务器后端尚未配置 API Key，请联系管理员！")
                else:
                    with st.spinner("AI 专家正在深度对比和分析您的方案，请稍候（可能需要 30-60 秒）..."):
                        evaluation_result = evaluate_document(
                            bidding_text, 
                            plan_text, 
                            api_key=BACKEND_API_KEY, 
                            base_url=BACKEND_BASE_URL, 
                            model=BACKEND_MODEL_NAME
                        )
                        
                    st.subheader("📊 评估报告与修改建议")
                    st.markdown(evaluation_result)
                    
                    # 提供下载报告功能
                    st.download_button(
                        label="📥 下载评估报告",
                        data=evaluation_result,
                        file_name="招标方案评估报告.md",
                        mime="text/markdown"
                    )
    elif bidding_file is not None or plan_file is not None:
        st.info("ℹ️ 请上传另一份文件以继续评估流程。")

if __name__ == "__main__":
    main()