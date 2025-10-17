# integrated_app.py
"""
完整的合同管理系统
集成用户认证、文件管理、智能缓存
"""

import streamlit as st
import sqlite3
import bcrypt
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Any
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

# LangChain相关导入
from langchain_rag_system import AdvancedContractRAG

# ===========================
# 数据库管理类
# ===========================

class DatabaseManager:
    """统一的数据库管理"""
    
    def __init__(self, db_path: str = "contract_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化所有数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                usage_count INTEGER DEFAULT 0,
                tier TEXT DEFAULT 'free'
            )
        """)
        
        # 处理过的文件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                file_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                processed_path TEXT NOT NULL,
                vector_store_path TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                file_hash TEXT,
                num_chunks INTEGER,
                num_pages INTEGER,
                processing_status TEXT DEFAULT 'pending',
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # 缓存的总结表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cached_summaries (
                summary_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                summary_type TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER,
                cost REAL,
                FOREIGN KEY (file_id) REFERENCES processed_files (file_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # 问答历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_history (
                qa_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens_used INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # 提取的信息缓存表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_info_cache (
                cache_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                extracted_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES processed_files (file_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        conn.commit()
        conn.close()

# ===========================
# 用户管理类
# ===========================

class UserManager:
    """用户认证和管理"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def register_user(self, username: str, email: str, password: str) -> Dict:
        """注册新用户"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        try:
            # 检查用户名和邮箱
            cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", 
                         (username, email))
            if cursor.fetchone():
                return {"success": False, "message": "用户名或邮箱已存在"}
            
            # 创建用户
            user_id = hashlib.md5(f"{username}_{datetime.now()}".encode()).hexdigest()[:16]
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cursor.execute("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, email, password_hash))
            
            # 创建用户目录结构
            user_dir = Path(f"user_data/{user_id}")
            user_dir.mkdir(parents=True, exist_ok=True)
            (user_dir / "contracts").mkdir(exist_ok=True)
            (user_dir / "vector_stores").mkdir(exist_ok=True)
            (user_dir / "cache").mkdir(exist_ok=True)
            
            conn.commit()
            return {"success": True, "user_id": user_id}
            
        except Exception as e:
            return {"success": False, "message": str(e)}
        finally:
            conn.close()
    
    def login(self, username: str, password: str) -> Dict:
        """用户登录"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT user_id, password_hash, email 
                FROM users 
                WHERE username = ? AND is_active = 1
            """, (username,))
            
            user = cursor.fetchone()
            if not user:
                return {"success": False}
            
            user_id, password_hash, email = user
            
            if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                # 更新登录时间和使用次数
                cursor.execute("""
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP, 
                        usage_count = usage_count + 1
                    WHERE user_id = ?
                """, (user_id,))
                conn.commit()
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "username": username,
                    "email": email
                }
            
            return {"success": False}
            
        finally:
            conn.close()

# ===========================
# 文件处理和缓存管理
# ===========================

class FileProcessor:
    """文件处理和缓存管理"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def process_and_save_file(self, user_id: str, uploaded_file, rag_system: AdvancedContractRAG) -> Dict:
        """处理并保存上传的文件"""
        
        # ⭐ 关键修改1: 在处理新文件前,先清理旧数据
        print(f"🧹 Clearing previous contract data before processing new file...")
        rag_system.clear_all_documents()
        
        # 生成文件ID
        file_id = hashlib.md5(
            f"{user_id}_{uploaded_file.name}_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        # 用户目录
        user_dir = Path(f"user_data/{user_id}")
        contracts_dir = user_dir / "contracts"
        vector_dir = user_dir / "vector_stores"
        
        contracts_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始文件
        file_path = contracts_dir / f"{file_id}_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 计算文件哈希
        file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()
        
        # 使用RAG系统处理文件
        result = rag_system.load_pdf(str(file_path), use_cache=True)
        
        # ⭐ 修复: 检查result是否为None
        if result is None:
            return {"success": False, "error": "load_pdf returned None - check RAG system"}
        
        if result.get("success", False):
            # 保存向量存储
            vector_store_path = vector_dir / f"{file_id}_vectors"
            rag_system.save_vectorstore(str(vector_store_path))
            
            # ⭐ 验证当前加载的文档
            current_docs = rag_system.get_current_documents_info()
            print(f"📋 Current documents after processing:\n{current_docs}")
            
            # 保存到数据库
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            stats = result.get("stats", {})
            cursor.execute("""
                INSERT INTO processed_files 
                (file_id, user_id, original_filename, processed_path, vector_store_path,
                 file_hash, num_chunks, num_pages, processing_status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?)
            """, (
                file_id,
                user_id,
                uploaded_file.name,
                str(file_path),
                str(vector_store_path),
                file_hash,
                stats.get("chunks", 0),
                stats.get("pages", 0),
                json.dumps(stats)
            ))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "file_id": file_id,
                "stats": stats
            }
        
        return {"success": False, "error": result.get("error", "Processing failed")}
    
    def get_recent_files(self, user_id: str, limit: int = 5) -> List[Dict]:
        """获取最近的文件"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT file_id, original_filename, upload_time, num_chunks, num_pages, 
                   processing_status, last_accessed
            FROM processed_files
            WHERE user_id = ? AND processing_status = 'completed'
            ORDER BY COALESCE(last_accessed, upload_time) DESC
            LIMIT ?
        """, (user_id, limit))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                "file_id": row[0],
                "filename": row[1],
                "upload_time": row[2],
                "num_chunks": row[3],
                "num_pages": row[4],
                "status": row[5],
                "last_accessed": row[6]
            })
        
        conn.close()
        return files
    
    def load_processed_file(self, user_id: str, file_id: str, rag_system: AdvancedContractRAG) -> bool:
        """加载已处理的文件到RAG系统"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT processed_path, vector_store_path, original_filename
            FROM processed_files
            WHERE file_id = ? AND user_id = ?
        """, (file_id, user_id))
        
        result = cursor.fetchone()
        if result:
            processed_path, vector_store_path, filename = result
            
            # 更新访问时间
            cursor.execute("""
                UPDATE processed_files 
                SET last_accessed = CURRENT_TIMESTAMP 
                WHERE file_id = ?
            """, (file_id,))
            conn.commit()
            
            try:
                # ⭐ 关键修改2: 彻底清理之前的所有数据
                print(f"🧹 Clearing all previous data before loading new contract...")
                rag_system.clear_all_documents()  # 使用专门的清理方法
                
                # ⭐ 关键修改3: 强制清空对话记忆,避免上下文混淆
                if hasattr(rag_system, 'memory') and rag_system.memory:
                    rag_system.memory.clear()
                    print(f"🧹 Cleared conversation memory")
                
                # 加载新的向量存储
                if vector_store_path and Path(vector_store_path).exists():
                    print(f"📂 Loading vector store for: {filename}")
                    # 这是安全的,因为我们加载的是自己创建的文件
                    rag_system.load_vectorstore(vector_store_path, allow_dangerous_deserialization=True)
                    
                    # ⭐ 关键修改4: 重新加载文档到内存(确保文档列表正确)
                    load_result = rag_system.load_pdf(processed_path, use_cache=True)
                    if load_result["success"]:
                        # ⭐ 关键修改5: 验证当前加载的文档
                        current_docs = rag_system.get_current_documents_info()
                        print(f"✅ Successfully loaded: {filename}")
                        print(f"📋 Current documents:\n{current_docs}")
                        conn.close()
                        return True
                    else:
                        print(f"⚠️ Failed to load document: {load_result.get('error')}")
                else:
                    # 如果没有向量存储,重新处理文件
                    print(f"🔄 No vector store found, reprocessing file...")
                    load_result = rag_system.load_pdf(processed_path, use_cache=False)
                    if load_result["success"]:
                        rag_system.save_vectorstore(vector_store_path)
                        print(f"✅ Reprocessed and loaded: {filename}")
                        conn.close()
                        return True
                    
            except Exception as e:
                print(f"❌ Error loading file: {e}")
                # 尝试重新处理
                try:
                    print(f"🔄 Attempting to reprocess from scratch...")
                    rag_system.clear_all_documents()  # 确保清理
                    rag_system.load_pdf(processed_path, use_cache=False)
                    rag_system.save_vectorstore(vector_store_path)
                    print(f"✅ Successfully reprocessed: {filename}")
                    conn.close()
                    return True
                except Exception as e2:
                    print(f"❌ Failed to reprocess: {e2}")
        
        conn.close()
        return False

# ===========================
# 缓存管理
# ===========================

class CacheManager:
    """管理各种缓存"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_cached_summary(self, file_id: str, summary_type: str) -> Optional[str]:
        """获取缓存的总结"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT summary_text 
            FROM cached_summaries
            WHERE file_id = ? AND summary_type = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (file_id, summary_type))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def save_summary(self, file_id: str, user_id: str, summary_type: str, 
                     summary_text: str, tokens_used: int = 0) -> None:
        """保存总结到缓存"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        summary_id = hashlib.md5(
            f"{file_id}_{summary_type}_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        # 删除旧的同类型总结
        cursor.execute("""
            DELETE FROM cached_summaries
            WHERE file_id = ? AND summary_type = ?
        """, (file_id, summary_type))
        
        # 保存新总结
        cursor.execute("""
            INSERT INTO cached_summaries
            (summary_id, file_id, user_id, summary_type, summary_text, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (summary_id, file_id, user_id, summary_type, summary_text, tokens_used))
        
        conn.commit()
        conn.close()
    
    def get_cached_extraction(self, file_id: str) -> Optional[Dict]:
        """获取缓存的信息提取结果"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT extracted_data
            FROM extracted_info_cache
            WHERE file_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (file_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return json.loads(result[0]) if result else None
    
    def save_extraction(self, file_id: str, user_id: str, extracted_data: Dict) -> None:
        """保存信息提取结果"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cache_id = hashlib.md5(
            f"{file_id}_extraction_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        # 删除旧的提取结果
        cursor.execute("DELETE FROM extracted_info_cache WHERE file_id = ?", (file_id,))
        
        # 保存新结果
        cursor.execute("""
            INSERT INTO extracted_info_cache
            (cache_id, file_id, user_id, extracted_data)
            VALUES (?, ?, ?, ?)
        """, (cache_id, file_id, user_id, json.dumps(extracted_data)))
        
        conn.commit()
        conn.close()
    
    def save_qa_history(self, user_id: str, file_id: str, question: str, 
                       answer: str, sources: List = None) -> None:
        """保存问答历史"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        qa_id = hashlib.md5(
            f"{user_id}_{question}_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        cursor.execute("""
            INSERT INTO qa_history
            (qa_id, user_id, file_id, question, answer, sources)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (qa_id, user_id, file_id, question, answer, json.dumps(sources)))
        
        conn.commit()
        conn.close()

# ===========================
# 主应用类
# ===========================

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
                                with st.expander("📚 来源"):
                                    for source in response["sources"]:
                                        st.write(f"• 页面 {source.get('page', 'N/A')}: {source.get('content', '')[:100]}...")
                            
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

# ===========================
# 主程序入口
# ===========================

if __name__ == "__main__":
    app = ContractAssistantApp()
    app.run()