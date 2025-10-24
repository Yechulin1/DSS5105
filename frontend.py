# frontend.py
"""
前端界面层
使用 Streamlit 构建用户界面
负责所有页面渲染和用户交互
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
load_dotenv()

# 导入后端类
from backend import (
    DatabaseManager,
    UserManager,
    FileProcessor,
    CacheManager
)

# 导入 RAG 系统
from langchain_rag_system import AdvancedContractRAG

# ==================================================
# 前端界面类
# ==================================================

class ContractAssistantApp:
    """主应用程序"""
    
    def __init__(self):
        # 初始化管理器
        self.db_manager = DatabaseManager()
        self.user_manager = UserManager(self.db_manager)
        self.file_processor = FileProcessor(self.db_manager)
        self.cache_manager = CacheManager(self.db_manager)
        
        # 初始化session state
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
        """登录页面"""
        st.title("📄 Contract Assistant - Login")
        
        tab1, tab2 = st.tabs(["登录", "注册"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("用户名")
                password = st.text_input("密码", type="password")
                submitted = st.form_submit_button("登录")
                
                if submitted:
                    result = self.user_manager.login(username, password)
                    if result["success"]:
                        st.session_state.authenticated = True
                        st.session_state.user_id = result["user_id"]
                        st.session_state.username = result["username"]
                        st.success("登录成功!")
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("用户名")
                new_email = st.text_input("邮箱")
                new_password = st.text_input("密码", type="password")
                confirm_password = st.text_input("确认密码", type="password")
                submitted = st.form_submit_button("注册")
                
                if submitted:
                    if new_password != confirm_password:
                        st.error("两次密码输入不一致")
                    elif len(new_password) < 6:
                        st.error("密码至少6位")
                    else:
                        result = self.user_manager.register_user(
                            new_username, new_email, new_password
                        )
                        if result["success"]:
                            st.success("注册成功!请登录")
                        else:
                            st.error(result.get("message", "注册失败"))
    
    def init_user_rag_system(self):
        """初始化用户的RAG系统"""
        if st.session_state.rag_system is None:
            st.session_state.rag_system = AdvancedContractRAG(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            )
            # 设置用户专属缓存目录
            user_cache_dir = Path(f"user_data/{st.session_state.user_id}/cache")
            user_cache_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.rag_system.cache_dir = user_cache_dir
    
    def main_app(self):
        """主应用界面"""
        st.set_page_config(page_title="Contract Assistant", page_icon="📄", layout="wide")
        
        # 初始化RAG系统
        self.init_user_rag_system()
        
        # 侧边栏
        with st.sidebar:
            st.write(f"👤 用户: **{st.session_state.username}**")
            
            if st.button("退出登录"):
                # ⭐ 关键修改6: 退出时清理RAG系统
                if st.session_state.rag_system:
                    st.session_state.rag_system.clear_all_documents()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            st.divider()
            
            # 显示最近的文件
            st.subheader("📁 最近的文件")
            recent_files = self.file_processor.get_recent_files(st.session_state.user_id)
            
            if recent_files:
                for file in recent_files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {file['filename'][:20]}...")
                    with col2:
                        if st.button("加载", key=f"load_{file['file_id']}"):
                            if self.file_processor.load_processed_file(
                                st.session_state.user_id,
                                file['file_id'],
                                st.session_state.rag_system
                            ):
                                st.session_state.current_file_id = file['file_id']
                                # ⭐ 关键修改7: 切换文件时清空聊天历史
                                st.session_state.messages = []
                                st.success("文件已加载")
                                st.rerun()
                    
                    # 显示文件信息
                    with st.expander(f"详情"):
                        st.write(f"页数: {file['num_pages']}")
                        st.write(f"分块数: {file['num_chunks']}")
                        st.write(f"上传时间: {file['upload_time']}")
            else:
                st.info("还没有上传文件")
        
        # 主界面
        st.title("📄 智能合同助手")
        
        # 当前加载的文件信息栏
        current_file_info = None
        if st.session_state.current_file_id:
            # 获取当前文件的详细信息
            for file in recent_files:
                if file['file_id'] == st.session_state.current_file_id:
                    current_file_info = file
                    break
            
            if current_file_info:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.success(f"📄 当前文件: **{current_file_info['filename']}**")
                with col2:
                    st.info(f"页数: {current_file_info['num_pages']}")
                with col3:
                    if st.button("🔄 切换文件"):
                        st.session_state.current_file_id = None
                        st.session_state.messages = []  # 清空聊天历史
                        # ⭐ 关键修改8: 切换文件时清理RAG系统
                        st.session_state.rag_system.clear_all_documents()
                        st.rerun()
            else:
                st.info(f"当前文件ID: {st.session_state.current_file_id}")
        else:
            st.warning("📂 请从左侧选择或上传一个文件")
        
        # 标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 上传", "💬 问答", "📝 总结", "🔍 提取", "📊 对比"
        ])
        
        # Tab1: 上传
        with tab1:
            uploaded_file = st.file_uploader("上传合同 (PDF)", type=['pdf'])
            
            if uploaded_file:
                if st.button("处理文件"):
                    with st.spinner("处理中..."):
                        result = self.file_processor.process_and_save_file(
                            st.session_state.user_id,
                            uploaded_file,
                            st.session_state.rag_system
                        )
                        
                        if result["success"]:
                            st.session_state.current_file_id = result["file_id"]
                            # ⭐ 关键修改9: 上传新文件时清空聊天历史
                            st.session_state.messages = []
                            st.success("文件处理完成!")
                            
                            # 显示统计
                            stats = result.get("stats", {})
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("页数", stats.get("pages", 0))
                            with col2:
                                st.metric("分块数", stats.get("chunks", 0))
                            with col3:
                                st.metric("字符数", f"{stats.get('characters', 0):,}")
                        else:
                            st.error(result.get("error", "处理失败"))
        
        # Tab2: 问答
        with tab2:
            if not st.session_state.current_file_id:
                st.warning("请先上传或加载一个文件")
            else:
                # ⭐ 新增: 显示当前正在使用的合同信息
                if current_file_info:
                    st.info(f"🎯 当前问答针对的合同: **{current_file_info['filename']}**")
                
                # ⭐ 新增: 显示当前RAG系统加载的文档信息(调试用)
                if st.checkbox("🔍 显示系统状态(调试)", value=False):
                    try:
                        rag_info = st.session_state.rag_system.get_current_documents_info()
                        st.code(rag_info)
                        
                        # 显示统计信息
                        stats = st.session_state.rag_system.get_statistics()
                        st.json(stats)
                    except Exception as e:
                        st.error(f"无法获取系统状态: {e}")
                
                # 聊天界面 - 显示历史消息
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        # 如果有来源信息,显示在展开框中
                        if message.get("sources"):
                            with st.expander("📚 来源"):
                                for source in message["sources"]:
                                    st.write(f"• {source}")
                
                # 输入框
                if prompt := st.chat_input("关于合同的问题..."):
                    # ⭐ 关键修改10: 在回答前再次验证文档状态
                    try:
                        current_docs = st.session_state.rag_system.get_current_documents_info()
                        if not current_docs or current_docs == "No documents loaded":
                            st.error("❌ 系统错误: 没有加载的文档,请重新加载合同")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ 文档验证失败: {e}")
                        st.stop()
                    
                    # 立即显示用户问题
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # 显示助手正在思考
                    with st.chat_message("assistant"):
                        with st.spinner("思考中..."):
                            response = st.session_state.rag_system.ask_question(prompt)
                            
                            # 保存到历史
                            self.cache_manager.save_qa_history(
                                st.session_state.user_id,
                                st.session_state.current_file_id,
                                prompt,
                                response["answer"],
                                response.get("sources", [])
                            )
                            
                            # 显示答案
                            st.write(response["answer"])
                            
                            # 显示来源
                            if response.get("sources"):
                                with st.expander("📚 来源参考", expanded=True):
                                    for i, source in enumerate(response["sources"], 1):
                                        st.markdown(f"**📄 来源 {i} - 页面 {source.get('page', 'N/A')}**")
                                        
                                        content = source.get('content', '')
                                        
                                        # 显示预览（前500字符）
                                        preview_length = 500
                                        if len(content) <= preview_length:
                                            st.text_area(
                                                f"来源内容_{i}",
                                                content,
                                                height=150,
                                                key=f"source_preview_{i}",
                                                label_visibility="collapsed"
                                            )
                                        else:
                                            # 显示预览
                                            st.text_area(
                                                f"来源内容预览_{i}",
                                                content[:preview_length] + "...",
                                                height=150,
                                                key=f"source_preview_{i}",
                                                label_visibility="collapsed"
                                            )
                                            
                                            # 提供查看完整内容的选项
                                            with st.expander(f"🔍 查看完整内容 ({len(content)} 字符)"):
                                                st.text_area(
                                                    f"完整内容_{i}",
                                                    content,
                                                    height=300,
                                                    key=f"source_full_{i}",
                                                    label_visibility="collapsed"
                                                )
                                        
                                        if i < len(response["sources"]):
                                            st.divider()
                            #------
                            # 保存助手消息到历史
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response["answer"],
                                "sources": response.get("sources", [])
                            })
                
                # 清除聊天历史按钮
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🗑️ 清除对话"):
                        st.session_state.messages = []
                        # ⭐ 关键修改11: 同时清除RAG系统的记忆
                        if hasattr(st.session_state.rag_system, 'memory'):
                            st.session_state.rag_system.memory.clear()
                        st.rerun()
        
        # Tab3: 总结
        with tab3:
            if not st.session_state.current_file_id:
                st.warning("请先上传或加载一个文件")
            else:
                summary_type = st.selectbox(
                    "总结类型",
                    ["brief", "comprehensive", "key_points"]
                )
                
                if st.button("生成总结"):
                    # 先检查缓存
                    cached = self.cache_manager.get_cached_summary(
                        st.session_state.current_file_id,
                        summary_type
                    )
                    
                    if cached:
                        st.success("使用缓存的总结")
                        st.write(cached)
                    else:
                        with st.spinner("生成总结中..."):
                            summary = st.session_state.rag_system.summarize_contract(
                                summary_type=summary_type
                            )
                            
                            # 保存到缓存
                            self.cache_manager.save_summary(
                                st.session_state.current_file_id,
                                st.session_state.user_id,
                                summary_type,
                                summary
                            )
                            
                            st.write(summary)
        
        # Tab4: 信息提取
        with tab4:
            if not st.session_state.current_file_id:
                st.warning("请先上传或加载一个文件")
            else:
                if st.button("提取关键信息"):
                    # 检查缓存
                    cached = self.cache_manager.get_cached_extraction(
                        st.session_state.current_file_id
                    )
                    
                    if cached:
                        st.success("使用缓存的提取结果")
                        key_info = cached
                    else:
                        with st.spinner("提取中..."):
                            key_info = st.session_state.rag_system.extract_key_information()
                            
                            # 保存到缓存
                            self.cache_manager.save_extraction(
                                st.session_state.current_file_id,
                                st.session_state.user_id,
                                key_info
                            )
                    
                    # 显示结果
                    df = pd.DataFrame([
                        {"字段": k, "值": v} for k, v in key_info.items()
                    ])
                    st.dataframe(df, use_container_width=True)
        
        # Tab5: 对比(简化版)
        with tab5:
            st.info("请加载两个文件进行对比")
            
            # 获取所有已处理的文件
            all_files = self.file_processor.get_recent_files(st.session_state.user_id, limit=20)
            
            if len(all_files) < 2:
                st.warning("至少需要2个文件才能进行对比")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    file1_options = {f['file_id']: f['filename'] for f in all_files}
                    file1_id = st.selectbox("选择文件1", options=list(file1_options.keys()), 
                                           format_func=lambda x: file1_options[x])
                
                with col2:
                    file2_options = {f['file_id']: f['filename'] for f in all_files if f['file_id'] != file1_id}
                    if file2_options:
                        file2_id = st.selectbox("选择文件2", options=list(file2_options.keys()), 
                                               format_func=lambda x: file2_options[x])
                    else:
                        st.warning("请选择不同的文件")
                        file2_id = None
                
                if file1_id and file2_id and st.button("开始对比"):
                    st.info("对比功能开发中... 需要加载两份合同进行分析")
    
    def run(self):
        """运行应用"""
        if st.session_state.authenticated:
            self.main_app()
        else:
            self.login_page()

