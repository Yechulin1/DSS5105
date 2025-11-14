# frontend.py
# frontend.py
"""
Frontend Interface Layer
Build user interface using Streamlit
Responsible for all page rendering and user interaction
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

# Import backend classes
from backend import (
    DatabaseManager,
    UserManager,
    FileProcessor,
    CacheManager
)

# Import RAG system
from langchain_rag_system import AdvancedContractRAG

# ==================================================
# Frontend Interface Class
# ==================================================
import re

def _render_markdown_safe(text: str):
    """在 Markdown 渲染前安全转义，使 $ 不触发 LaTeX。"""
    if not isinstance(text, str):
        text = str(text)
    # 先保护真正的 $$...$$ 数学块（很少见，但以防万一）
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: r'\\$\\$' + m.group(1) + r'\\$\\$', text, flags=re.S)
    # 再把所有单个 $ 转义为 \$
    text = text.replace('$', r'\$')
    # 用 markdown 渲染（不用 unsafe_allow_html）
    import streamlit as st
    st.markdown(text)

class ContractAssistantApp:
    """Main application"""
    
    def __init__(self):
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.user_manager = UserManager(self.db_manager)
        self.file_processor = FileProcessor(self.db_manager)
        self.cache_manager = CacheManager(self.db_manager)
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'current_file_id' not in st.session_state:
            st.session_state.current_file_id = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
    
    def login_page(self):
        """Login page"""
        st.title("📄 Contract Assistant - Login")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    result = self.user_manager.login(username, password)
                    if result["success"]:
                        st.session_state.authenticated = True
                        st.session_state.user_id = result["user_id"]
                        st.session_state.username = result["username"]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        result = self.user_manager.register_user(
                            new_username, new_email, new_password
                        )
                        if result["success"]:
                            st.success("Registration successful! Please login")
                        else:
                            st.error(result.get("message", "Registration failed"))
    
    def init_user_rag_system(self):
        """Initialize user's RAG system"""
        if st.session_state.rag_system is None:
            st.session_state.rag_system = AdvancedContractRAG(
                api_key = os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            )
            # Set user-specific cache directory
            user_cache_dir = Path(f"user_data/{st.session_state.user_id}/cache")
            user_cache_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.rag_system.cache_dir = user_cache_dir
    
    def main_app(self):
        """Main application interface"""
        st.set_page_config(page_title="Contract Assistant", page_icon="📄", layout="wide")
        
        # Initialize RAG system
        self.init_user_rag_system()
        
        # Sidebar
        with st.sidebar:
            st.write(f"👤 User: **{st.session_state.username}**")
            
            if st.button("Logout"):
                # ⭐ Key modification 6: Clean up RAG system on logout
                if st.session_state.rag_system:
                    st.session_state.rag_system.clear_all_documents()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            st.divider()
            
            # Display recent files
            st.subheader("📁 Recent Files")
            recent_files = self.file_processor.get_recent_files(st.session_state.user_id)
            
            if recent_files:
                for file in recent_files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {file['filename'][:20]}...")
                    with col2:
                        if st.button("Load", key=f"load_{file['file_id']}"):
                            if self.file_processor.load_processed_file(
                                st.session_state.user_id,
                                file['file_id'],
                                st.session_state.rag_system
                            ):
                                st.session_state.current_file_id = file['file_id']
                                # ⭐ Key modification 7: Clear chat history when switching files
                                st.session_state.messages = []
                                st.success("File loaded")
                                st.rerun()
                    
                    # Display file information
                    with st.expander(f"Details"):
                        st.write(f"Pages: {file['num_pages']}")
                        st.write(f"Chunks: {file['num_chunks']}")
                        st.write(f"Upload time: {file['upload_time']}")
            else:
                st.info("No files uploaded yet")
        
        # Main interface
        st.title("📄 Intelligent Contract Assistant")
        
        # Current loaded file information bar
        current_file_info = None
        if st.session_state.current_file_id:
            # Get detailed information about current file
            for file in recent_files:
                if file['file_id'] == st.session_state.current_file_id:
                    current_file_info = file
                    break
            
            if current_file_info:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.success(f"📄 Current file: **{current_file_info['filename']}**")
                with col2:
                    st.info(f"Pages: {current_file_info['num_pages']}")
                with col3:
                    if st.button("🔄 Switch File"):
                        st.session_state.current_file_id = None
                        st.session_state.messages = []  # Clear chat history
                        # ⭐ Key modification 8: Clean RAG system when switching files
                        st.session_state.rag_system.clear_all_documents()
                        st.rerun()
            else:
                st.info(f"Current file ID: {st.session_state.current_file_id}")
        else:
            st.warning("📂 Please select or upload a file from the left sidebar")
        
        # Create tabs for different functions
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 Upload", "💬 Q&A", "📝 Summary", "🔍 Extract Info", "⚖️ Compare"
        ])
        
        # Tab1: Upload
        with tab1:
            uploaded_file = st.file_uploader("Upload Contract (PDF)", type=['pdf'])
            
            if uploaded_file:
                if st.button("Start Processing"):
                    with st.spinner("Processing file..."):
                        result = self.file_processor.process_and_save_file(
                            st.session_state.user_id,
                            uploaded_file,
                            st.session_state.rag_system
                        )
                        
                        if result["success"]:
                            st.session_state.current_file_id = result["file_id"]
                            # ⭐ Key modification 9: Clear chat history when uploading new file
                            st.session_state.messages = []
                            st.success("File processed successfully!")
                            
                            # Display processing information
                            stats = result.get("stats", {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total pages", stats.get("pages", 0))
                            with col2:
                                st.metric("Total chunks", stats.get("chunks", 0))
                            with col3:
                                st.metric("File size", f"{stats.get('characters', 0):,}")
                        else:
                            st.error(result.get("error", "Processing failed"))
        
        # Tab2: Q&A
        with tab2:
            if not st.session_state.current_file_id:
                st.warning("Please upload or load a file first")
            else:
                # ⭐ New: Display current contract information in use
                if current_file_info:
                    st.info(f"🎯 Current Q&A for contract: **{current_file_info['filename']}**")
                
                
                """ # ⭐ 新增: 显示当前RAG系统加载的文档信息(调试用)
                if st.checkbox("🔍 system debugging", value=False):
                    try:
                        rag_info = st.session_state.rag_system.get_current_documents_info()
                        st.code(rag_info)
                        
                        # Display processing information信息
                        stats = st.session_state.rag_system.get_statistics()
                        st.json(stats)
                    except Exception as e:
                        st.error(f"无法获取系统状态: {e}") """
                
                # Chat interface - Display chat history
                for msg_idx, message in enumerate(st.session_state.messages):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        # Display sources if available (same format as new messages)
                        if message.get("sources"):
                            with st.expander("📚 Reference Sources"):
                                for i, source in enumerate(message["sources"], 1):
                                    page_number = source.get('page', 'N/A')
                                    if page_number is not None and isinstance(page_number, int):
                                        page_number += 1  # 将页面编号从 0 改为 1 开始
                                    else:
                                        page_number = 'N/A'
                                    st.markdown(f"**📄 Source {i} - Page {page_number}**")
                                    
                                    content = source.get('content', '')
                                    
                                    # Display preview (first 300 characters)
                                    preview_length = 300
                                    if len(content) <= preview_length:
                                        st.text_area(
                                            f"Source content",
                                            content,
                                            height=100,
                                            key=f"hist_src_{msg_idx}_{i}",
                                            label_visibility="collapsed"
                                        )
                                    else:
                                        # Display preview with expand option
                                        st.text_area(
                                            f"Source preview",
                                            content[:preview_length] + "...",
                                            height=100,
                                            key=f"hist_prev_{msg_idx}_{i}",
                                            label_visibility="collapsed"
                                        )
                                        
                                        with st.expander(f"🔍 View Full Content ({len(content)} characters)"):
                                            st.text_area(
                                                f"Full content",
                                                content,
                                                height=300,
                                                key=f"hist_full_{msg_idx}_{i}",
                                                label_visibility="collapsed"
                                            )
                                    
                                    if i < len(message["sources"]):
                                        st.divider()
                
                # Chat input
                # Add disclaimer below the chat input
                st.caption("AI can make mistakes. Please verify important information.")
                
                if prompt := st.chat_input("Ask a question about the contract..."):
                    # ⭐ Key modification 10: Validate document status before answering
                    try:
                        current_docs = st.session_state.rag_system.get_current_documents_info()
                        if not current_docs or current_docs == "No documents loaded":
                            st.error("❌ System error: No documents loaded, please reload the contract")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ 文档验证失败: {e}")
                        st.stop()
                    
                    # Display user question immediately
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Display assistant thinking
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = st.session_state.rag_system.ask_question(prompt)
                            
                            # Save to history
                            self.cache_manager.save_qa_history(
                                st.session_state.user_id,
                                st.session_state.current_file_id,
                                prompt,
                                response["answer"],
                                response.get("sources", [])
                            )
                            
                            # Display answer
                            st.write(response["answer"])
                            
                            # Display sources
                            if response.get("sources"):
                                with st.expander("📚 Reference Sources", expanded=True):
                                    for i, source in enumerate(response["sources"], 1):
                                        st.markdown(f"**📄 Source {i} - Page {source.get('page', 'N/A')}**")
                                        
                                        content = source.get('content', '')
                                        
                                        # Display preview (first 500 characters)
                                        preview_length = 500
                                        if len(content) <= preview_length:
                                            st.text_area(
                                                f"Source content_{i}",
                                                content,
                                                height=150,
                                                key=f"new_source_preview_{len(st.session_state.messages)}_{i}",  # ← 加上消息计数
                                                label_visibility="collapsed"
                                            )
                                        else:
                                            # Display preview
                                            st.text_area(
                                                f"Source content preview_{i}",
                                                content[:preview_length] + "...",
                                                height=150,
                                                key=f"new_source_preview_long_{len(st.session_state.messages)}_{i}",  # ← 唯一key
                                                label_visibility="collapsed"
                                            )
                                            # Provide option to view full content
                                            with st.expander(f"🔍 View full content ({len(content)} Characters)"):
                                                st.text_area(
                                                        f"Full content_{i}",
                                                        content,
                                                        height=300,
                                                        key=f"new_source_full_{len(st.session_state.messages)}_{i}",  # ← 唯一key
                                                        label_visibility="collapsed"
                                                    )
                                        
                                        if i < len(response["sources"]):
                                            st.divider()
                            #------
                            # Save assistant message to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response.get("sources", [])
                            })
                
                # Clear chat history button
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.messages = []
                        # ⭐ Key modification 11: Also clear RAG system's memory
                        if hasattr(st.session_state.rag_system, 'memory'):
                            st.session_state.rag_system.memory.clear()
                        st.rerun()
        
        # Tab3: Summary
        with tab3:
            if not st.session_state.current_file_id:
                st.warning("Please upload or load a file first")
            else:
                summary_type = st.selectbox(
                    "Summary Type",
                    ["brief", "comprehensive", "key_points"]
                )
                
                if st.button("Generate Summary"):
                    # Check cache first
                    cached = self.cache_manager.get_cached_summary(
                        st.session_state.current_file_id,
                        summary_type
                    )
                    
                    if cached:
                        st.success("Using cached summary")
                        st.write(cached)
                    else:
                        with st.spinner("Generating summary..."):
                            summary = st.session_state.rag_system.summarize_contract(
                                summary_type=summary_type
                            )
                            
                            # Save to cache
                            self.cache_manager.save_summary(
                                st.session_state.current_file_id,
                                st.session_state.user_id,
                                summary_type,
                                summary
                            )
                            
                            _render_markdown_safe(summary)

        
        # Tab4: 信息Extract Info
        with tab4:
            if not st.session_state.current_file_id:
                st.warning("Please upload or load a file first")
            else:
                if st.button("Extract Key Information"):
                    # 检查缓存
                    cached = self.cache_manager.get_cached_extraction(
                        st.session_state.current_file_id
                    )
                    
                    if cached:
                        st.success("Using cached extraction results")
                        key_info = cached
                    else:
                        with st.spinner("Extracting..."):
                            key_info = st.session_state.rag_system.extract_key_information_parallel()
                            
                            # Save to cache
                            self.cache_manager.save_extraction(
                                st.session_state.current_file_id,
                                st.session_state.user_id,
                                key_info
                            )
                    
                    # Display results
                    df = pd.DataFrame([
                        {"Field": k, "Value": v} for k, v in key_info.items()
                    ])
                    st.dataframe(df, use_container_width=True)
        
        # Tab5: Compare(简化版)
        with tab5:
            st.info("Load two files to compare")
            
            # Get all processed files
            all_files = self.file_processor.get_recent_files(st.session_state.user_id, limit=20)
            
            if len(all_files) < 2:
                st.warning("At least 2 files are required for comparison")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    file1_options = {f['file_id']: f['filename'] for f in all_files}
                    file1_id = st.selectbox("Select File 1", options=list(file1_options.keys()), 
                                           format_func=lambda x: file1_options[x])
                
                with col2:
                    file2_options = {f['file_id']: f['filename'] for f in all_files if f['file_id'] != file1_id}
                    if file2_options:
                        file2_id = st.selectbox("Select File 2", options=list(file2_options.keys()), 
                                               format_func=lambda x: file2_options[x])
                    else:
                        st.warning("Please select different files")
                        file2_id = None
                
                if file1_id and file2_id and st.button("Start Comparison"):
                    st.info("Comparison feature under development... Need to load two contracts for analysis")
    
    def run(self):
        """Run application"""
        if st.session_state.authenticated:
            self.main_app()
        else:
            self.login_page()