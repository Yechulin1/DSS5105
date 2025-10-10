# app_with_auth.py
"""带用户认证的完整应用"""

import streamlit as st
from auth_manager import AuthenticatedApp
from langchain_streamlit_app import main as original_main
from pathlib import Path

class IntegratedApp(AuthenticatedApp):
    """集成原有功能的认证应用"""
    
    def main_app(self):
        """覆盖主应用方法，集成原有功能"""
        # 显示用户信息
        with st.sidebar:
            st.write(f"👤 User: **{st.session_state['username']}**")
            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # 设置用户专属的RAG系统
        if 'rag_system' not in st.session_state:
            st.session_state['rag_system'] = st.session_state.get('user_rag_system')
        
        # 运行原来的主应用
        original_main()

if __name__ == "__main__":
    app = IntegratedApp()
    app.run()