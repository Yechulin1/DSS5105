# frontend_beautiful.py
"""
前端界面层 - 美化版
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
# 自定义CSS样式
# ==================================================

CUSTOM_CSS = """
<style>
    /* 主题色变量 */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
        --bg-light: #f8f9fa;
        --bg-dark: #2c3e50;
    }
    
    /* 全局样式优化 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 主容器 */
    .main .block-container {
        padding: 2rem 3rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        max-width: 1400px;
    }
    
    /* 标题样式 */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    /* 卡片样式 */
    .css-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .css-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* 按钮美化 */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* 输入框美化 */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* 文件上传器美化 */
    [data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: #f0f0ff;
    }
    
    /* 标签页美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background: #f0f0f0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* 成功/错误/警告消息美化 */
    .stSuccess {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        border: none;
    }
    
    /* 指标卡片美化 */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* 聊天消息美化 */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    [data-testid="stChatMessageContent"] {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* 展开器美化 */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 600;
    }
    
    /* 分隔线美化 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* 加载动画 */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* 侧边栏按钮特殊样式 */
    [data-testid="stSidebar"] .stButton>button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 2px solid white;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: white;
        color: #667eea;
    }
    
    /* 表单美化 */
    .stForm {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        border: 2px solid #e0e0e0;
    }
    
    /* 代码块美化 */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* 自定义动画 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
"""

# ==================================================
# 前端界面类
# ==================================================

class ContractAssistantApp:
    """主应用程序 - 美化版"""
    
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
        """登录页面 - 美化版"""
        # 应用自定义CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # 页面配置
        st.set_page_config(
            page_title="智能合同助手 - 登录",
            page_icon="📄",
            layout="centered",
            initial_sidebar_state="collapsed"
        )
        
        # Logo和标题
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style='text-align: center; padding: 2rem 0;'>
                    <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>📄</h1>
                    <h1 style='font-size: 2rem; margin: 0;'>智能合同助手</h1>
                    <p style='color: #7f8c8d; margin-top: 0.5rem;'>Contract Assistant Powered by AI</p>
                </div>
            """, unsafe_allow_html=True)
        
        # 登录/注册表单
        tab1, tab2 = st.tabs(["🔐 登录", "📝 注册"])
        
        with tab1:
            st.markdown("### 欢迎回来")
            with st.form("login_form"):
                username = st.text_input("👤 用户名", placeholder="请输入用户名")
                password = st.text_input("🔒 密码", type="password", placeholder="请输入密码")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    submitted = st.form_submit_button("🚀 登录", use_container_width=True)
                
                if submitted:
                    if not username or not password:
                        st.error("❌ 请填写所有字段")
                    else:
                        with st.spinner("🔄 验证中..."):
                            result = self.user_manager.login(username, password)
                            if result["success"]:
                                st.session_state.authenticated = True
                                st.session_state.user_id = result["user_id"]
                                st.session_state.username = result["username"]
                                st.success(f"✅ 欢迎回来，{result['username']}！")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ 用户名或密码错误")
        
        with tab2:
            st.markdown("### 创建新账户")
            with st.form("register_form"):
                new_username = st.text_input("👤 用户名", placeholder="请输入用户名")
                new_email = st.text_input("📧 邮箱", placeholder="请输入邮箱地址")
                new_password = st.text_input("🔒 密码", type="password", placeholder="至少6位字符")
                confirm_password = st.text_input("🔒 确认密码", type="password", placeholder="再次输入密码")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    submitted = st.form_submit_button("✨ 注册", use_container_width=True)
                
                if submitted:
                    if not all([new_username, new_email, new_password, confirm_password]):
                        st.error("❌ 请填写所有字段")
                    elif new_password != confirm_password:
                        st.error("❌ 两次密码输入不一致")
                    elif len(new_password) < 6:
                        st.error("❌ 密码至少需要6位字符")
                    else:
                        with st.spinner("🔄 注册中..."):
                            result = self.user_manager.register_user(
                                new_username, new_email, new_password
                            )
                            if result["success"]:
                                st.success("✅ 注册成功！请切换到登录标签")
                                st.balloons()
                            else:
                                st.error(f"❌ {result.get('message', '注册失败')}")
        
        # 页脚
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #95a5a6; font-size: 0.9rem;'>
                <p>💡 使用 AI 技术，让合同分析更简单</p>
                <p>🔒 您的数据安全受到保护</p>
            </div>
        """, unsafe_allow_html=True)
    
    def init_user_rag_system(self):
        """初始化用户的RAG系统"""
        if not st.session_state.get('rag_system'):
            st.session_state.rag_system = AdvancedContractRAG(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            )
            
            # 设置用户专属缓存目录
            user_cache_dir = Path(f"user_data/{st.session_state.user_id}/cache")
            user_cache_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.rag_system.cache_dir = user_cache_dir
    
    def main_app(self):
        """主应用界面 - 美化版"""
        # 应用自定义CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # 页面配置
        st.set_page_config(
            page_title="智能合同助手",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 初始化RAG系统
        self.init_user_rag_system()
        
        # 侧边栏美化
        with st.sidebar:
            # 用户信息卡片
            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; backdrop-filter: blur(10px);'>
                    <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                        <div style='background: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; font-size: 1.5rem;'>
                            👤
                        </div>
                        <div>
                            <h3 style='color: white; margin: 0; font-size: 1.2rem;'>{st.session_state.username}</h3>
                            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>会员用户</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 退出登录", use_container_width=True):
                # 清理RAG系统
                if st.session_state.rag_system:
                    st.session_state.rag_system.clear_all_documents()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
            
            # 最近文件
            st.markdown("### 📁 最近的文件")
            recent_files = self.file_processor.get_recent_files(st.session_state.user_id, limit=5)
            
            if recent_files:
                for file in recent_files:
                    # 文件卡片
                    is_current = file['file_id'] == st.session_state.current_file_id
                    bg_color = "rgba(255,255,255,0.3)" if is_current else "rgba(255,255,255,0.1)"
                    
                    st.markdown(f"""
                        <div style='background: {bg_color}; padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; backdrop-filter: blur(10px);'>
                            <p style='color: white; margin: 0; font-weight: 600; font-size: 0.9rem;'>📄 {file['filename'][:25]}...</p>
                            <p style='color: rgba(255,255,255,0.8); margin: 0.3rem 0 0 0; font-size: 0.8rem;'>
                                📊 {file['num_pages']}页 · 🧩 {file['num_chunks']}块
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if not is_current:
                        if st.button("📂 加载", key=f"load_{file['file_id']}", use_container_width=True):
                            if self.file_processor.load_processed_file(
                                st.session_state.user_id,
                                file['file_id'],
                                st.session_state.rag_system
                            ):
                                st.session_state.current_file_id = file['file_id']
                                st.session_state.messages = []
                                st.success("✅ 文件已加载")
                                st.rerun()
                    else:
                        st.success("✅ 当前文件")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("📭 还没有上传文件")
        
        # 主界面
        # 顶部标题栏
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0 2rem 0;'>
                <h1 style='font-size: 2.5rem; margin: 0;'>📄 智能合同助手</h1>
                <p style='color: #7f8c8d; margin: 0.5rem 0 0 0;'>Contract Assistant · AI-Powered Analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 当前文件信息栏（美化版）
        current_file_info = None
        if st.session_state.current_file_id:
            recent_files = self.file_processor.get_recent_files(st.session_state.user_id)
            for file in recent_files:
                if file['file_id'] == st.session_state.current_file_id:
                    current_file_info = file
                    break
            
            if current_file_info:
                # 美化的文件信息卡片
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; color: white;
                                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h3 style='margin: 0; color: white;'>📄 {current_file_info['filename']}</h3>
                                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                                    📊 {current_file_info['num_pages']} 页 · 
                                    🧩 {current_file_info['num_chunks']} 个文档块 · 
                                    📅 {current_file_info['upload_time']}
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 切换文件", key="switch_file"):
                    st.session_state.current_file_id = None
                    st.session_state.messages = []
                    st.session_state.rag_system.clear_all_documents()
                    st.rerun()
            else:
                st.info(f"📋 当前文件ID: {st.session_state.current_file_id}")
        else:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
                            padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; color: white;
                            text-align: center; box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3);'>
                    <h3 style='margin: 0; color: white;'>📂 请从左侧选择或上传一个文件</h3>
                    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>开始你的智能合同分析之旅</p>
                </div>
            """, unsafe_allow_html=True)
        
        # 标签页（美化版图标）
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 上传合同", 
            "💬 智能问答", 
            "📝 生成摘要", 
            "🔍 信息提取", 
            "📊 合同对比"
        ])
        
        # Tab1: 上传
        with tab1:
            st.markdown("### 📤 上传新合同")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "选择PDF文件",
                    type=['pdf'],
                    help="支持PDF格式，最大100MB"
                )
            
            with col2:
                st.markdown("""
                    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-top: 1.5rem;'>
                        <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>📋 支持格式</h4>
                        <p style='margin: 0; font-size: 0.9rem;'>✓ PDF文档</p>
                        <p style='margin: 0; font-size: 0.9rem;'>✓ 最大100MB</p>
                        <p style='margin: 0; font-size: 0.9rem;'>✓ 中英文支持</p>
                    </div>
                """, unsafe_allow_html=True)
            
            if uploaded_file:
                st.success(f"✅ 已选择文件: {uploaded_file.name}")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🚀 开始处理", use_container_width=True, type="primary"):
                        with st.spinner("🔄 正在处理文件，请稍候..."):
                            result = self.file_processor.process_and_save_file(
                                st.session_state.user_id,
                                uploaded_file,
                                st.session_state.rag_system
                            )
                            
                            if result["success"]:
                                st.session_state.current_file_id = result["file_id"]
                                st.session_state.messages = []
                                st.success("✅ 文件处理完成！")
                                st.balloons()
                                
                                # 显示统计信息（美化版）
                                stats = result.get("stats", {})
                                st.markdown("### 📊 文件统计")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📄 页数", stats.get("pages", 0))
                                with col2:
                                    st.metric("🧩 文档块", stats.get("chunks", 0))
                                with col3:
                                    st.metric("📝 字符数", f"{stats.get('characters', 0):,}")
                                with col4:
                                    st.metric("💾 大小", f"{stats.get('size_mb', 0):.1f}MB")
                            else:
                                st.error(f"❌ {result.get('error', '处理失败')}")
        
        # Tab2: 问答
        with tab2:
            if not st.session_state.current_file_id:
                st.markdown("""
                    <div style='text-align: center; padding: 3rem;'>
                        <h2 style='color: #7f8c8d;'>💬</h2>
                        <h3 style='color: #7f8c8d;'>请先上传或加载一个文件</h3>
                        <p style='color: #95a5a6;'>开始智能问答之旅</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # 当前文件提示
                if current_file_info:
                    st.info(f"🎯 当前问答针对: **{current_file_info['filename']}**")
                
                # 显示系统状态（可选）
                with st.expander("🔍 系统状态（调试）", expanded=False):
                    try:
                        rag_info = st.session_state.rag_system.get_current_documents_info()
                        st.code(rag_info, language="text")
                        
                        stats = st.session_state.rag_system.get_statistics()
                        st.json(stats)
                    except Exception as e:
                        st.error(f"无法获取系统状态: {e}")
                
                # 聊天界面
                st.markdown("### 💬 对话历史")
                
                # 显示历史消息
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "sources" in message and message["sources"]:
                            with st.expander("📚 参考来源"):
                                for i, source in enumerate(message["sources"], 1):
                                    st.markdown(f"**来源 {i}:**")
                                    st.text(source[:200] + "...")
                
                # 聊天输入
                if prompt := st.chat_input("💭 输入你的问题..."):
                    # 添加用户消息
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # 生成回答
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 思考中..."):
                            response = st.session_state.rag_system.ask_question(prompt)
                            
                            if response.get("success"):
                                answer = response["answer"]
                                sources = response.get("sources", [])
                                
                                st.markdown(answer)
                                
                                if sources:
                                    with st.expander("📚 参考来源"):
                                        for i, source in enumerate(sources, 1):
                                            st.markdown(f"**来源 {i}:**")
                                            st.text(source[:200] + "...")
                                
                                # 保存到历史
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": sources
                                })
                                
                                # 保存到数据库
                                if st.session_state.current_file_id:
                                    self.cache_manager.save_qa_history(
                                        st.session_state.user_id,
                                        st.session_state.current_file_id,
                                        prompt,
                                        answer,
                                        sources
                                    )
                            else:
                                error_msg = f"❌ {response.get('error', '未知错误')}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                
                # 清空对话按钮
                if st.session_state.messages:
                    if st.button("🗑️ 清空对话历史"):
                        st.session_state.messages = []
                        st.rerun()
        
        # Tab3: 总结
        with tab3:
            if not st.session_state.current_file_id:
                st.markdown("""
                    <div style='text-align: center; padding: 3rem;'>
                        <h2 style='color: #7f8c8d;'>📝</h2>
                        <h3 style='color: #7f8c8d;'>请先上传或加载一个文件</h3>
                        <p style='color: #95a5a6;'>生成智能摘要</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 📝 生成合同摘要")
                
                # 摘要类型选择
                col1, col2 = st.columns([1, 2])
                with col1:
                    summary_type = st.selectbox(
                        "选择摘要类型",
                        ["comprehensive", "brief", "key_points"],
                        format_func=lambda x: {
                            "comprehensive": "📋 全面摘要（详细版）",
                            "brief": "📄 简短摘要（精简版）",
                            "key_points": "🔑 关键点（列表版）"
                        }[x]
                    )
                
                with col2:
                    st.markdown("""
                        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                            <p style='margin: 0; font-size: 0.9rem;'><strong>全面摘要:</strong> 包含所有重要条款和细节</p>
                            <p style='margin: 0.3rem 0; font-size: 0.9rem;'><strong>简短摘要:</strong> 1-2段核心内容概括</p>
                            <p style='margin: 0; font-size: 0.9rem;'><strong>关键点:</strong> 结构化的要点列表</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                if st.button("✨ 生成摘要", use_container_width=True, type="primary"):
                    # 先检查缓存
                    cached = self.cache_manager.get_cached_summary(
                        st.session_state.current_file_id,
                        summary_type
                    )
                    
                    if cached:
                        st.success("✅ 使用缓存的摘要")
                        st.markdown("### 📄 摘要内容")
                        st.markdown(f"""
                            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                                        border-left: 4px solid #667eea;'>
                                {cached}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        with st.spinner("🔄 正在生成摘要..."):
                            summary = st.session_state.rag_system.summarize_contract(
                                summary_type=summary_type
                            )
                            
                            if summary and "Error" not in summary:
                                st.success("✅ 摘要生成完成！")
                                st.markdown("### 📄 摘要内容")
                                st.markdown(f"""
                                    <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                                                border-left: 4px solid #667eea;'>
                                        {summary}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # 保存到缓存
                                self.cache_manager.save_summary(
                                    st.session_state.current_file_id,
                                    st.session_state.user_id,
                                    summary_type,
                                    summary
                                )
                            else:
                                st.error(f"❌ 生成失败: {summary}")
        
        # Tab4: 提取
        with tab4:
            if not st.session_state.current_file_id:
                st.markdown("""
                    <div style='text-align: center; padding: 3rem;'>
                        <h2 style='color: #7f8c8d;'>🔍</h2>
                        <h3 style='color: #7f8c8d;'>请先上传或加载一个文件</h3>
                        <p style='color: #95a5a6;'>提取关键信息</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 🔍 信息提取")
                st.info("💡 自动提取合同中的关键信息字段")
                
                if st.button("🚀 开始提取", use_container_width=True, type="primary"):
                    # 检查缓存
                    cached = self.cache_manager.get_cached_extraction(
                        st.session_state.current_file_id
                    )
                    
                    if cached:
                        st.success("✅ 使用缓存的提取结果")
                        st.json(cached)
                    else:
                        with st.spinner("🔄 正在提取信息..."):
                            extracted = st.session_state.rag_system.extract_contract_info()
                            
                            if extracted:
                                st.success("✅ 提取完成！")
                                
                                # 美化显示
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### 📋 基本信息")
                                    if "parties" in extracted:
                                        st.info(f"👥 当事人: {extracted['parties']}")
                                    if "rental_amount" in extracted:
                                        st.info(f"💰 租金: {extracted['rental_amount']}")
                                    if "lease_duration" in extracted:
                                        st.info(f"📅 租期: {extracted['lease_duration']}")
                                
                                with col2:
                                    st.markdown("#### 💵 费用信息")
                                    if "deposit" in extracted:
                                        st.info(f"💎 押金: {extracted['deposit']}")
                                    if "additional_fees" in extracted:
                                        st.info(f"➕ 其他费用: {extracted['additional_fees']}")
                                
                                # 完整JSON
                                with st.expander("📄 完整JSON数据"):
                                    st.json(extracted)
                                
                                # 保存缓存
                                self.cache_manager.save_extraction(
                                    st.session_state.current_file_id,
                                    st.session_state.user_id,
                                    extracted
                                )
                            else:
                                st.error("❌ 提取失败")
        
        # Tab5: 对比
        with tab5:
            st.markdown("### 📊 合同对比")
            st.markdown("""
                <div style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                            padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; color: white;
                            text-align: center;'>
                    <h3 style='margin: 0; color: white;'>🚧 功能开发中</h3>
                    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>即将支持多份合同的对比分析</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 对比功能将支持：条款差异、价格对比、风险评估等")
    
    def run(self):
        """运行应用"""
        if st.session_state.authenticated:
            self.main_app()
        else:
            self.login_page()