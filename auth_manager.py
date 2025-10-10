# auth_manager.py
"""
用户认证和文件管理系统
支持用户注册、登录、文件历史管理
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import bcrypt
from pathlib import Path
import json
from datetime import datetime
import shutil
from typing import Dict, List, Optional
import sqlite3
import hashlib

class UserManager:
    """用户管理系统 - 使用SQLite数据库"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表"""
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
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # 文件记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_files (
                file_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # 用户会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_user(self, username: str, email: str, password: str) -> Dict:
        """注册新用户"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 检查用户名和邮箱是否已存在
            cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                return {"success": False, "message": "Username or email already exists"}
            
            # 生成用户ID和密码哈希
            user_id = hashlib.md5(f"{username}_{datetime.now()}".encode()).hexdigest()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # 插入用户
            cursor.execute("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            """, (user_id, username, email, password_hash))
            
            # 创建用户文件夹
            user_folder = Path(f"user_data/{user_id}")
            user_folder.mkdir(parents=True, exist_ok=True)
            (user_folder / "contracts").mkdir(exist_ok=True)
            (user_folder / "cache").mkdir(exist_ok=True)
            
            conn.commit()
            return {"success": True, "message": "Registration successful", "user_id": user_id}
            
        except Exception as e:
            return {"success": False, "message": f"Registration failed: {str(e)}"}
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Dict:
        """验证用户登录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 获取用户信息
            cursor.execute("SELECT user_id, password_hash FROM users WHERE username = ? AND is_active = 1", (username,))
            user = cursor.fetchone()
            
            if not user:
                return {"success": False, "message": "Invalid username or password"}
            
            user_id, password_hash = user
            
            # 验证密码
            if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                # 更新最后登录时间
                cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (user_id,))
                conn.commit()
                
                return {"success": True, "user_id": user_id, "username": username}
            else:
                return {"success": False, "message": "Invalid username or password"}
                
        except Exception as e:
            return {"success": False, "message": f"Authentication failed: {str(e)}"}
        finally:
            conn.close()
    
    def save_user_file(self, user_id: str, file_path: str, metadata: Dict = None) -> Dict:
        """保存用户上传的文件记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            file_path = Path(file_path)
            file_id = hashlib.md5(f"{user_id}_{file_path.name}_{datetime.now()}".encode()).hexdigest()
            
            # 计算文件哈希
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # 移动文件到用户目录
            user_dir = Path(f"user_data/{user_id}/contracts")
            user_dir.mkdir(parents=True, exist_ok=True)
            
            new_file_path = user_dir / f"{file_id}_{file_path.name}"
            shutil.copy2(file_path, new_file_path)
            
            # 保存到数据库
            cursor.execute("""
                INSERT INTO user_files (file_id, user_id, filename, file_path, file_size, file_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                user_id,
                file_path.name,
                str(new_file_path),
                file_path.stat().st_size,
                file_hash,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            return {"success": True, "file_id": file_id, "file_path": str(new_file_path)}
            
        except Exception as e:
            return {"success": False, "message": f"Failed to save file: {str(e)}"}
        finally:
            conn.close()
    
    def get_user_files(self, user_id: str) -> List[Dict]:
        """获取用户的所有文件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT file_id, filename, file_size, upload_time, metadata
                FROM user_files
                WHERE user_id = ?
                ORDER BY upload_time DESC
            """, (user_id,))
            
            files = []
            for row in cursor.fetchall():
                files.append({
                    "file_id": row[0],
                    "filename": row[1],
                    "file_size": row[2],
                    "upload_time": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {}
                })
            
            return files
            
        except Exception as e:
            print(f"Error getting user files: {e}")
            return []
        finally:
            conn.close()
    
    def get_file_path(self, user_id: str, file_id: str) -> Optional[str]:
        """获取文件路径"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT file_path FROM user_files
                WHERE user_id = ? AND file_id = ?
            """, (user_id, file_id))
            
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error getting file path: {e}")
            return None
        finally:
            conn.close()

class AuthenticatedApp:
    """带认证功能的主应用"""
    
    def __init__(self):
        self.user_manager = UserManager()
        
        # 初始化session state
        if 'authentication_status' not in st.session_state:
            st.session_state['authentication_status'] = None
        if 'username' not in st.session_state:
            st.session_state['username'] = None
        if 'user_id' not in st.session_state:
            st.session_state['user_id'] = None
        if 'user_rag_system' not in st.session_state:
            st.session_state['user_rag_system'] = None
    
    def login_page(self):
        """登录页面"""
        st.markdown("## 🔐 Login to Contract Assistant")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    result = self.user_manager.authenticate_user(username, password)
                    if result["success"]:
                        st.session_state['authentication_status'] = True
                        st.session_state['username'] = result["username"]
                        st.session_state['user_id'] = result["user_id"]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result["message"])
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Register")
                
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        result = self.user_manager.register_user(new_username, new_email, new_password)
                        if result["success"]:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(result["message"])
    
    def main_app(self):
        """主应用界面（登录后）"""
        # 侧边栏用户信息
        with st.sidebar:
            st.write(f"👤 Logged in as: **{st.session_state['username']}**")
            if st.button("Logout"):
                for key in ['authentication_status', 'username', 'user_id', 'user_rag_system']:
                    st.session_state[key] = None
                st.rerun()
            
            st.divider()
            
            # 显示用户文件历史
            st.subheader("📁 Your Files")
            user_files = self.user_manager.get_user_files(st.session_state['user_id'])
            
            if user_files:
                for file in user_files:
                    with st.expander(file['filename']):
                        st.write(f"Uploaded: {file['upload_time']}")
                        st.write(f"Size: {file['file_size']:,} bytes")
                        if st.button(f"Load", key=f"load_{file['file_id']}"):
                            # 加载文件到RAG系统
                            file_path = self.user_manager.get_file_path(
                                st.session_state['user_id'],
                                file['file_id']
                            )
                            if file_path:
                                st.session_state['selected_file'] = file_path
                                st.success(f"Loaded: {file['filename']}")
            else:
                st.info("No files uploaded yet")
        
        # 初始化用户的RAG系统
        if st.session_state['user_rag_system'] is None:
            try:
                from config import OPENAI_API_KEY, OPENAI_MODEL
                from langchain_rag_system import AdvancedContractRAG
                
                # 为每个用户创建独立的RAG实例
                st.session_state['user_rag_system'] = AdvancedContractRAG(
                    api_key=OPENAI_API_KEY,
                    model=OPENAI_MODEL
                )
                
                # 设置用户专属的缓存目录
                user_cache_dir = Path(f"user_data/{st.session_state['user_id']}/cache")
                user_cache_dir.mkdir(parents=True, exist_ok=True)
                st.session_state['user_rag_system'].cache_dir = user_cache_dir
                
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                return
        
        # 在这里集成原来的主应用功能
        # 导入并运行原来的app，但使用用户专属的RAG系统
        st.title("📄 Your Contract Assistant")
        
        # 文件上传（保存到用户目录）
        uploaded_file = st.file_uploader("Upload Contract (PDF)", type=['pdf'])
        if uploaded_file:
            # 保存到临时位置
            temp_path = Path("temp") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Process and Save"):
                # 保存到用户目录
                save_result = self.user_manager.save_user_file(
                    st.session_state['user_id'],
                    str(temp_path),
                    metadata={"original_name": uploaded_file.name}
                )
                
                if save_result["success"]:
                    # 加载到RAG系统
                    load_result = st.session_state['user_rag_system'].load_pdf(
                        save_result['file_path']
                    )
                    
                    if load_result["success"]:
                        st.success("Contract processed and saved!")
                        st.rerun()
                    else:
                        st.error(f"Failed to process: {load_result.get('error')}")
    
    def run(self):
        """运行应用"""
        st.set_page_config(
            page_title="Contract Assistant",
            page_icon="📄",
            layout="wide"
        )
        
        if st.session_state['authentication_status']:
            self.main_app()
        else:
            self.login_page()

if __name__ == "__main__":
    app = AuthenticatedApp()
    app.run()