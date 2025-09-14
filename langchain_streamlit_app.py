"""
高级合同管理系统 - Streamlit界面（修正版）
修复点：
1) 避免未定义的 cache_key 访问，使用 st.session_state.last_summary_key
2) 聊天提问处更稳健的异常处理与提示
3) 其它逻辑与原版一致

依赖：langchain_rag_system.py（已在同目录或 Python 路径下）
"""

import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

from langchain_rag_system import AdvancedContractRAG

# -----------------------------
# 页面配置
# -----------------------------
st.set_page_config(
    page_title="📄 Advanced Contract Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 自定义CSS
# -----------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .source-box {
            background-color: #e8f4f8;
            padding: 0.5rem;
            border-left: 3px solid #1f77b4;
            margin: 0.5rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session State 初始化
# -----------------------------
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_contracts" not in st.session_state:
    st.session_state.loaded_contracts = []
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}
# 用于避免 cache_key 未定义访问
if "last_summary_key" not in st.session_state:
    st.session_state.last_summary_key = None

# -----------------------------
# 系统初始化
# -----------------------------
def initialize_system() -> bool:
    """初始化 RAG 系统，读取 config 中的 OPENAI_API_KEY / OPENAI_MODEL。"""
    try:
        from config import OPENAI_API_KEY, OPENAI_MODEL
    except ImportError:
        st.error("❌ config.py not found. Please create it from config.example.py")
        return False

    try:
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
            st.session_state.rag_system = AdvancedContractRAG(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
            )
            return True
        else:
            st.error("⚠️ Please configure your OpenAI API key in config.py")
            return False
    except Exception as e:
        st.error(f"❌ Error initializing system: {e}")
        return False

# -----------------------------
# 主体
# -----------------------------
def main():
    # 标题
    st.markdown(
        '<h1 class="main-header">📄 Advanced Contract Management System</h1>',
        unsafe_allow_html=True,
    )

    # 初始化系统
    if st.session_state.rag_system is None:
        with st.spinner("🔄 Initializing AI system..."):
            if initialize_system():
                st.success("✅ System initialized successfully!")
            else:
                st.stop()

    # 标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Manage",
        "💬 Q&A Chat",
        "📝 Summarize",
        "🔍 Extract Info",
        "📊 Compare",
    ])

    # -------------------------
    # Tab 1: 上传和管理
    # -------------------------
    with tab1:
        st.header("📤 Contract Upload & Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload Contract (PDF only)",
                type=["pdf"],
                help="Upload your rental contract in PDF format",
            )

            if uploaded_file is not None:
                # 保存上传文件
                upload_path = Path("uploads")
                upload_path.mkdir(exist_ok=True)
                file_path = upload_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if st.button("🔄 Process Contract", type="primary"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            result = st.session_state.rag_system.load_pdf(str(file_path))
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {e}")
                            result = {"success": False, "error": str(e)}

                        if result.get("success"):
                            st.success(f"✅ {result['message']}")
                            if str(file_path) not in st.session_state.loaded_contracts:
                                st.session_state.loaded_contracts.append(str(file_path))

                            # 显示统计
                            stats = result.get("stats", {})
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Pages", stats.get("pages", 0))
                            with col_b:
                                st.metric("Chunks", stats.get("chunks", 0))
                            with col_c:
                                st.metric("Characters", f"{stats.get('characters', 0):,}")
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")

        with col2:
            st.subheader("📊 System Status")
            if st.session_state.rag_system:
                try:
                    stats = st.session_state.rag_system.get_statistics()
                except Exception as e:
                    st.error(f"❌ Failed to get statistics: {e}")
                    stats = {}

                st.metric("Loaded Contracts", stats.get("loaded_contracts", 0))
                st.metric("Total Chunks", stats.get("total_chunks", 0))
                st.metric("Vector Store Size", stats.get("vector_store_size", 0))

                contracts = stats.get("contracts", [])
                if contracts:
                    st.subheader("📑 Loaded Files")
                    for contract in contracts:
                        filename = contract.get("file", "(unknown)")
                        with st.expander(filename):
                            st.write(f"Pages: {contract.get('pages', '-')}")
                            st.write(f"Chunks: {contract.get('chunks', '-')}")
                            st.write(f"Loader: {contract.get('loader', '-')}")

    # -------------------------
    # Tab 2: 问答聊天
    # -------------------------
    with tab2:
        st.header("💬 Contract Q&A Chat")

        if not st.session_state.loaded_contracts:
            st.warning("⚠️ Please upload a contract first in the 'Upload & Manage' tab")
        else:
            chat_container = st.container()

            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if "sources" in message and message["sources"]:
                            with st.expander("📚 Sources"):
                                for source in message["sources"]:
                                    st.markdown(
                                        f"""
                                        <div class=\"source-box\">\n
                                        <strong>File:</strong> {Path(source.get('source','?')).name}<br>
                                        <strong>Page:</strong> {source.get('page','?')}<br>
                                        <strong>Content:</strong> {source.get('content','')}
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

            # 快速问题
            st.subheader("💡 Quick Questions")
            col_q1, col_q2, col_q3 = st.columns(3)
            quick_questions = [
                "What is the monthly rent?",
                "When is rent due?",
                "What's the security deposit?",
                "Can I have pets?",
                "How to terminate early?",
                "Who handles maintenance?",
            ]
            for i, q in enumerate(quick_questions):
                col = [col_q1, col_q2, col_q3][i % 3]
                if col.button(q, key=f"quick_{i}"):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

            # 聊天输入
            if prompt := st.chat_input("Ask about your contract..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_system.ask_question(prompt)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response.get("answer", "(no answer)"),
                                "sources": response.get("sources", []),
                            }
                        )
                    except Exception as e:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"❌ Error answering your question: {e}",
                            }
                        )
                st.rerun()

    # -------------------------
    # Tab 3: 合同总结（修复 cache_key 未定义）
    # -------------------------
    with tab3:
        st.header("📝 Contract Summarization")

        if not st.session_state.loaded_contracts:
            st.warning("⚠️ Please upload a contract first")
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                # 选择总结类型
                summary_type = st.selectbox(
                    "Summary Type",
                    ["brief", "comprehensive", "key_points"],
                    format_func=lambda x: {
                        "brief": "📄 Brief Summary",
                        "comprehensive": "📚 Comprehensive Summary",
                        "key_points": "🎯 Key Points",
                    }[x],
                )

                # 选择要总结的合同
                if len(st.session_state.loaded_contracts) > 1:
                    contract_to_summarize = st.selectbox(
                        "Select Contract",
                        ["All Contracts"] + st.session_state.loaded_contracts,
                        format_func=lambda x: "All" if x == "All Contracts" else Path(x).name,
                    )
                else:
                    contract_to_summarize = (
                        st.session_state.loaded_contracts[0]
                        if st.session_state.loaded_contracts
                        else None
                    )

                if st.button("📝 Generate Summary", type="primary"):
                    with st.spinner("Generating summary..."):
                        cache_key = f"{contract_to_summarize}_{summary_type}"
                        pdf_path = (
                            None
                            if contract_to_summarize == "All Contracts"
                            else contract_to_summarize
                        )
                        try:
                            if cache_key in st.session_state.summary_cache:
                                summary = st.session_state.summary_cache[cache_key]
                            else:
                                summary = st.session_state.rag_system.summarize_contract(
                                    pdf_path=pdf_path,
                                    summary_type=summary_type,
                                )
                                st.session_state.summary_cache[cache_key] = summary

                            # 记录最后一次的 key，供右侧展示使用
                            st.session_state.last_summary_key = cache_key
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {e}")

            with col2:
                st.subheader("📋 Summary Result")
                key = st.session_state.get("last_summary_key")
                if key and key in st.session_state.summary_cache:
                    summary = st.session_state.summary_cache[key]
                    if summary_type == "key_points":
                        st.markdown(summary)
                    else:
                        st.write(summary)

                    st.download_button(
                        "📥 Download Summary",
                        summary,
                        file_name=f"contract_summary_{summary_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("📝 Please generate a summary first.")

    # -------------------------
    # Tab 4: 信息提取
    # -------------------------
    with tab4:
        st.header("🔍 Extract Contract Information")
        if not st.session_state.loaded_contracts:
            st.warning("⚠️ Please upload a contract first")
        else:
            if st.button("🔍 Extract All Key Information", type="primary"):
                with st.spinner("Extracting information..."):
                    try:
                        key_info = st.session_state.rag_system.extract_key_information()
                    except Exception as e:
                        key_info = {"error": str(e)}

                    if "error" not in key_info:
                        df = pd.DataFrame(
                            [
                                {"Field": k.replace("_", " ").title(), "Value": v}
                                for k, v in key_info.items()
                            ]
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        st.download_button(
                            "📥 Download as JSON",
                            json.dumps(key_info, indent=2),
                            file_name=f"contract_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )

                        st.subheader("📊 Contract Details")
                        colx, coly, colz = st.columns(3)
                        with colx:
                            st.info(f"**Monthly Rent**\n\n{key_info.get('rent_amount', 'N/A')}")
                            st.info(f"**Security Deposit**\n\n{key_info.get('security_deposit', 'N/A')}")
                            st.info(f"**Late Fee**\n\n{key_info.get('late_fee', 'N/A')}")
                        with coly:
                            st.info(f"**Lease Duration**\n\n{key_info.get('lease_duration', 'N/A')}")
                            st.info(f"**Payment Due**\n\n{key_info.get('payment_due_date', 'N/A')}")
                            st.info(f"**Utilities**\n\n{key_info.get('utilities', 'N/A')}")
                        with colz:
                            st.info(f"**Pet Policy**\n\n{key_info.get('pet_policy', 'N/A')}")
                            st.info(f"**Parking**\n\n{key_info.get('parking', 'N/A')}")
                            st.info(f"**Termination**\n\n{key_info.get('termination', 'N/A')}")
                    else:
                        st.error(key_info["error"])

    # -------------------------
    # Tab 5: 合同对比
    # -------------------------
    with tab5:
        st.header("📊 Compare Contracts")
        if len(st.session_state.loaded_contracts) < 2:
            st.warning("⚠️ Please upload at least 2 contracts to compare")
        else:
            col1, col2 = st.columns(2)
            with col1:
                contract1 = st.selectbox(
                    "Select First Contract",
                    st.session_state.loaded_contracts,
                    format_func=lambda x: Path(x).name,
                    key="compare_1",
                )
            with col2:
                contract2 = st.selectbox(
                    "Select Second Contract",
                    [c for c in st.session_state.loaded_contracts if c != contract1],
                    format_func=lambda x: Path(x).name,
                    key="compare_2",
                )

            if st.button("🔍 Compare Contracts", type="primary"):
                with st.spinner("Comparing contracts..."):
                    try:
                        comparison = st.session_state.rag_system.compare_contracts(
                            contract1, contract2
                        )
                        st.subheader("📋 Comparison Results")
                        st.markdown(comparison)
                        st.download_button(
                            "📥 Download Comparison",
                            comparison,
                            file_name=f"contract_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                        )
                    except Exception as e:
                        st.error(f"❌ Error comparing: {e}")

    # -------------------------
    # 侧边栏
    # -------------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        if st.button("🗑️ Clear Cache"):
            st.session_state.summary_cache = {}
            if st.session_state.rag_system:
                try:
                    st.session_state.rag_system.clear_memory()
                except Exception as e:
                    st.warning(f"⚠️ Failed to clear memory: {e}")
            st.session_state.last_summary_key = None
            st.success("Cache cleared!")

        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.success("Chat history cleared!")

        st.divider()
        st.subheader("💾 Vector Store")
        if st.button("💾 Save Vector Store"):
            if st.session_state.rag_system:
                try:
                    st.session_state.rag_system.save_vectorstore()
                    st.success("Vector store saved!")
                except Exception as e:
                    st.error(f"❌ Save failed: {e}")
        if st.button("📂 Load Vector Store"):
            if st.session_state.rag_system:
                try:
                    st.session_state.rag_system.load_vectorstore()
                    st.success("Vector store loaded!")
                except Exception as e:
                    st.error(f"❌ Load failed: {e}")

        st.divider()
        st.subheader("📖 How to Use")
        st.markdown(
            """
            1. **Upload**: Upload PDF contracts in the first tab  
            2. **Chat**: Ask questions about your contracts  
            3. **Summarize**: Generate different types of summaries  
            4. **Extract**: Get structured information  
            5. **Compare**: Compare multiple contracts  

            **Tips:**
            - Use cache to save API costs  
            - Save vector store for faster loading  
            - Clear chat history to start fresh  
            """
        )


if __name__ == "__main__":
    main()
