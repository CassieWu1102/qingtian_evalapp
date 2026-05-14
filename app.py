import streamlit as st
import docx
import PyPDF2
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载后端环境变量
load_dotenv()

# ================= 配置双模型环境 =================
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
KIMI_MODEL_NAME = os.getenv("KIMI_MODEL_NAME", "moonshot-v1-128k")
KIMI_MAX_TOKENS = int(os.getenv("KIMI_MAX_TOKENS", "4000"))

SPARK_API_KEY = os.getenv("SPARK_API_KEY", "")
SPARK_BASE_URL = os.getenv("SPARK_BASE_URL", "https://spark-api-open.xf-yun.com/v1")
SPARK_MODEL_NAME = os.getenv("SPARK_MODEL_NAME", "generalv3.5")
SPARK_MAX_TOKENS = int(os.getenv("SPARK_MAX_TOKENS", "4000"))

st.set_page_config(page_title="多模型智能评标与重写系统", page_icon="🤖", layout="wide")

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

# --- 2. AI 评估逻辑 (步骤1：获取单专家意见) ---
def get_expert_suggestions(bidding_text, plan_text, expert_type, api_key, base_url, model, max_tokens):
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    if expert_type == "kimi":
        system_prompt = """
        你是一位具备国际顶级视角的资深招标与评标专家（专家A）。
        请严格对比【招标文件】，对用户提交的【招标组织方案】进行深度分析。
        
        你的主要任务是：
        1. 🔍 **需求响应度审查**：检查方案是否有遗漏项、未响应项或矛盾点（废标风险）。
        2. 🧠 **逻辑与专业性审查**：方案的整体逻辑是否严密，操作流程是否具备实际可行性。
        3. 💡 **具体修改意见**：指出问题并提供专业的修改建议。
        
        注意：你只需要输出“问题排查和修改意见”，**不需要**输出完整重写后的方案。
        """
    else:
        system_prompt = """
        你是一位深谙国内招标法及科大讯飞“青天”大模型自动化评标系统规则的评标专家（专家B）。
        请严格对比【招标文件】，对用户提交的【招标组织方案】进行深度分析。
        
        你的主要任务是：
        1. 🤖 **AI 评标友好度**：方案是否采用了清晰的“点对点”响应结构？是否存在排版混乱导致 AI 误判或漏判的风险？
        2. 🔑 **核心关键词匹配**：方案中是否充分包含了得分关键词（如“公平、公正、公开”、“全流程追溯”、“应急预案”、“信息安全”、“保密管理”等）？
        3. ⚖️ **国内合规风险**：指出常被国内评标系统扣分的“硬伤”。
        4. 💡 **具体修改意见**：指出问题并提供针对 AI 评标系统的修改建议。
        
        注意：你只需要输出“问题排查和修改意见”，**不需要**输出完整重写后的方案。
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
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[{expert_type} 专家评估出错]: {str(e)}"

# --- 3. AI 评估逻辑 (步骤2：综合意见并重写) ---
def synthesize_and_rewrite(bidding_text, plan_text, gemini_feedback, spark_feedback, api_key, base_url, model, max_tokens):
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    system_prompt = """
    你是一位顶级的标书主编。你现在收到了两位资深评标专家对一份【招标组织方案】的修改意见：
    - 专家A（侧重需求响应与逻辑）：指出了结构和响应度方面的问题。
    - 专家B（侧重“青天”大模型评标规则）：指出了关键词、AI提取友好度和合规性问题。
    
    请你综合这两位专家的意见，结合原始的【招标文件】和【招标组织方案】，完成以下任务：
    
    1. 📈 **综合评估总结**：结合两位专家的意见，给出一个综合的预估分数（1-100分），并简短总结该方案的核心优缺点。
    2. 🛠️ **核心修改说明**：简述你在重写时，吸收了哪些关键建议进行了修改。
    3. ✨ **优化后的招标组织方案（完整版）**：
       - 这是最重要的部分！请输出一份**完整、可以直接复制使用**的最终方案文本。
       - 确保内容高度结构化，多用清晰的层级列表（如 1.1, 1.1.1）。
       - 完美融入专家B建议的核心得分关键词。
       - 严格响应专家A指出的所有遗漏项和逻辑漏洞。
    """
    
    user_content = f"【招标文件】\n{bidding_text}\n\n【原招标组织方案】\n{plan_text}\n\n====================\n\n【专家A意见】\n{gemini_feedback}\n\n【专家B意见】\n{spark_feedback}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[主编整合重写出错]: {str(e)}"

# --- 4. Streamlit 前端交互界面 ---
def main():
    st.title("🤖 双模型智能评标分析系统")
    st.markdown("""
    本系统采用 **Google Kimi** 与 **科大讯飞“青天”模型(星火)** 协同工作。
    由两位 AI 专家分别从**逻辑合规**和**AI评标系统偏好**进行双重审查，为您提供详尽的修改建议。
    """)
    
    with st.sidebar:
        st.header("⚙️ 后端系统状态")
        
        st.subheader("🌐 Kimi 引擎 (逻辑与响应度审查)")
        if KIMI_API_KEY:
            st.success(f"✅ 已连接: `{KIMI_MODEL_NAME}`")
        else:
            st.error("⚠️ 未配置 Kimi 密钥")
            
        st.subheader("🧠 讯飞青天引擎 (AI评标规则审查)")
        if SPARK_API_KEY:
            st.success(f"✅ 已连接: `{SPARK_MODEL_NAME}`")
        else:
            st.error("⚠️ 未配置讯飞星火密钥")
            
        st.markdown("---")
        st.info("💡 请联系系统管理员在服务器 `.env` 文件中配置以上密钥。")

    st.subheader("1. 📤 上传文档")
    st.info("💡 提示：对于 50-100 页（约数十万 Tokens）的超长方案，AI 处理需要较长时间。")
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
                
            st.subheader("2. 🚀 开始多模型联合评估")
            
            # 增加一个让用户选择是否需要重写的复选框
            need_synthesis = st.checkbox("✨ 启用主编模式：综合两位专家意见，并直接生成优化后的最终完整方案", value=False)
            
            if st.button("启动联合评估", type="primary"):
                if not KIMI_API_KEY or not SPARK_API_KEY:
                    st.warning("⚠️ 必须同时配置 Kimi 和 科大讯飞 API Key 才能启动联合评估！")
                else:
                    total_steps = 3 if need_synthesis else 2
                    
                    progress_text = f"步骤 1/{total_steps}: Kimi 专家正在进行深度逻辑审查..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    # Step 1: Kimi 意见
                    kimi_opinion = get_expert_suggestions(
                        bidding_text, plan_text, "kimi", 
                        KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL_NAME, KIMI_MAX_TOKENS
                    )
                    my_bar.progress(int(100/total_steps), text=f"步骤 2/{total_steps}: 科大讯飞专家正在进行 AI 评标规则审查...")
                    
                    # Step 2: Spark 意见
                    spark_opinion = get_expert_suggestions(
                        bidding_text, plan_text, "spark", 
                        SPARK_API_KEY, SPARK_BASE_URL, SPARK_MODEL_NAME, SPARK_MAX_TOKENS
                    )
                    
                    if need_synthesis:
                        my_bar.progress(int(200/total_steps), text=f"步骤 3/{total_steps}: 正在综合两位专家意见，重写最终完美方案...")
                        # Step 3: 综合重写
                        final_result = synthesize_and_rewrite(
                            bidding_text, plan_text, kimi_opinion, spark_opinion,
                            KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL_NAME, KIMI_MAX_TOKENS
                        )
                        my_bar.progress(100, text="✅ 评估与重写完成！")
                        
                        st.markdown("---")
                        st.subheader("📑 第一阶段：双专家独立意见")
                        with st.expander("查看 Kimi 专家的逻辑与响应度审查意见"):
                            st.markdown(kimi_opinion)
                        with st.expander("查看 讯飞青天专家的评标系统规则审查意见"):
                            st.markdown(spark_opinion)
                            
                        st.markdown("---")
                        st.subheader("🏆 第二阶段：最终综合评估与重写方案")
                        st.markdown(final_result)
                        
                        st.download_button(
                            label="📥 下载最终重写方案",
                            data=final_result,
                            file_name="双模型优化_最终招标方案.md",
                            mime="text/markdown"
                        )
                    else:
                        my_bar.progress(100, text="✅ 双专家评估完成！")
                        
                        st.markdown("---")
                        st.subheader("📑 Kimi 专家：逻辑与需求响应度审查")
                        st.markdown(kimi_opinion)
                            
                        st.markdown("---")
                        st.subheader("📑 讯飞青天专家：AI 评标规则与合规性审查")
                        st.markdown(spark_opinion)
                        
                        # 组合两者内容用于下载
                        combined_result = f"# Kimi 专家评估意见\n\n{kimi_opinion}\n\n---\n\n# 讯飞青天专家评估意见\n\n{spark_opinion}"
                        
                        st.download_button(
                            label="📥 下载双专家评估报告",
                            data=combined_result,
                            file_name="双专家招标方案评估报告.md",
                            mime="text/markdown"
                        )
    elif bidding_file is not None or plan_file is not None:
        st.info("ℹ️ 请上传另一份文件以继续流程。")

if __name__ == "__main__":
    main()
