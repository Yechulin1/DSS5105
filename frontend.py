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

# Custom CSS for quick question buttons
QUICK_QUESTION_CSS = """
<style>
    /* Quick question button styling */
    div[data-testid="column"] > div > div > button {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 12px 8px;
        font-size: 0.95rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: auto;
        min-height: 60px;
        white-space: normal;
        line-height: 1.3;
    }
    
    div[data-testid="column"] > div > div > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        border-color: #667eea;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    div[data-testid="column"] > div > div > button p {
        color: white !important;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Quick question section */
    .quick-questions-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
</style>
"""

# Marketing page CSS - copied from frontend_reference.py
HERO_CSS = """
<style>
  .marketing-hero {display:grid;grid-template-columns:1fr;gap:2.25rem;align-items:center;padding:1.25rem 0;}
  .gradient-primary {background:linear-gradient(90deg,#a78bfa 0%, #ff7e5f 50%, #feb47b 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;color:transparent;display:inline-block;}
  .bg-gradient-subtle {background: radial-gradient(700px circle at 18% 20%, rgba(167,139,250,0.10) 0%, transparent 50%), radial-gradient(800px circle at 85% 80%, rgba(255,154,98,0.12) 0%, transparent 52%), #ffffff;}
  .shadow-glow {box-shadow:0 14px 36px rgba(118,75,162,0.20);} 
  .marketing-card {background:#ffffff;border-radius:20px;border:1px solid #e6e6e6;min-height:420px;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;padding:2.25rem;}
  .social-avatars {display:flex;}
  .social-avatars .avatar {width:40px;height:40px;border-radius:50%;background:#ff9a62;border:2px solid #ffffff;margin-left:-10px;}
  .rating {display:flex;align-items:center;gap:8px;color:#7f8c8d;}
  .hero-title {font-size:clamp(7.2rem,14.4vw,11.6rem);line-height:1.02;font-weight:800;color:#2c3e50;margin:0.1rem 0 0.6rem;}
  .hero-desc {font-size:2.9rem;color:#4b5563;max-width:90ch;line-height:1.65;margin:0.25rem 0 0.95rem;}
  h1.hero-title {font-size:5rem !important; line-height:1.1 !important; margin-left:auto; margin-right:auto; white-space:normal; max-width:none; display:block;}
  p.hero-desc {font-size:1.5rem !important;}
  .cta-row {display:flex;gap:80px;flex-wrap:wrap;margin-top:1.1rem;}
  .cta {border-radius:16px;padding:18px 32px;font-weight:700;text-decoration:none!important;color:#fff;display:inline-flex;align-items:center;justify-content:center;min-width:300px;font-size:1.2rem;}
  .cta:hover {text-decoration:none!important;}
  .cta-primary {background:#ff7e5f;color:#fff;box-shadow:0 8px 26px rgba(255,126,95,0.28);} 
  .cta-outline {background:#ffffff;color:#111827;border:1.5px solid #d1d5db;box-shadow:0 6px 18px rgba(17,24,39,0.08);} 
  .cta-primary:link, .cta-primary:visited, .cta-primary:hover, .cta-primary:active { color:#fff !important; text-decoration:none !important; }
  .cta-outline:link, .cta-outline:visited, .cta-outline:hover, .cta-outline:active { color:#111827 !important; text-decoration:none !important; }
  .hero-card-icon {font-size:56px;color:#ff9a62;opacity:0.9;}
  .trust {display:flex;gap:22px;margin-top:28px;color:#6b7280;font-size:1.1rem;}
  .trust span {display:inline-block;min-width:260px;}
  .trust-line {height:1px;background:#e5e7eb;margin:18px 0 0 0;}
  .copyright {text-align:center;color:#9aa1a9;font-size:.95rem;margin-top:12px;}
  .container {max-width:1280px;margin:0 auto;padding:0 16px;}
  @keyframes floatGrad {0%{background-position:0% 0%,100% 100%;}50%{background-position:20% 10%,80% 90%;}100%{background-position:0% 0%,100% 100%;}}
  .animated-bg {background-image: radial-gradient(800px circle at 20% 20%, rgba(167,139,250,0.22) 0%, transparent 55%), radial-gradient(900px circle at 80% 75%, rgba(255,154,98,0.20) 0%, transparent 55%), linear-gradient(180deg,#ffffff,#ffffff);background-repeat:no-repeat;animation:floatGrad 12s ease-in-out infinite;}
  .hero-layer {position:fixed;inset:0;z-index:0;background-image: radial-gradient(1100px circle at 20% 20%, rgba(167,139,250,0.12) 0%, transparent 55%), radial-gradient(1100px circle at 80% 75%, rgba(255,154,98,0.12) 0%, transparent 55%), linear-gradient(180deg,#faf7ff,#ffffff);background-repeat:no-repeat;animation:floatGrad 16s ease-in-out infinite;}
  .hero-section {position:relative;z-index:1;padding:1.5rem 0 1rem;min-height:100vh;display:flex;align-items:center;justify-content:center;}
  .hero-container {width:80vw;max-width:1600px;margin:0 auto;padding:0 40px;}
  @keyframes flowLight {0%{transform:translate3d(-16%, -12%, 0) rotate(0deg); filter:hue-rotate(0deg);}50%{transform:translate3d(18%, 12%, 0) rotate(180deg); filter:hue-rotate(18deg);}100%{transform:translate3d(-16%, -12%, 0) rotate(360deg); filter:hue-rotate(0deg);}}
  .hero-layer::before {content:""; position:absolute; inset:-10%; pointer-events:none;
    background:
      radial-gradient(580px circle at 26% 24%, rgba(255,154,98,0.16) 0%, transparent 62%),
      conic-gradient(from 0deg at 50% 50%, rgba(167,139,250,0.12), rgba(255,154,98,0.14), rgba(255,255,255,0.0), rgba(167,139,250,0.12));
    filter: blur(26px);
    animation: flowLight 16s ease-in-out infinite;
    opacity: 0.85;
  }
  .hero-layer::after {content:""; position:absolute; inset:-12%; pointer-events:none;
    background:
      radial-gradient(520px circle at 74% 72%, rgba(167,139,250,0.12) 0%, transparent 62%),
      conic-gradient(from 120deg at 50% 50%, rgba(255,154,98,0.10), rgba(167,139,250,0.12), rgba(255,255,255,0.0), rgba(255,154,98,0.10));
    filter: blur(22px);
    animation: flowLight 20s ease-in-out infinite reverse;
    opacity: 0.80;
  }
  .marketing-hero > div { text-align:center; }
  .social-avatars { justify-content:center; }
  .rating { justify-content:center; }
  .cta-row { justify-content:center; }
  .trust { justify-content:center; }
</style>
"""

# Global theme CSS - copied from frontend_reference.py
GLOBAL_CSS = """
<style>
  @keyframes globalFloatGrad {0%{background-position:0% 0%,100% 100%;}50%{background-position:20% 10%,80% 90%;}100%{background-position:0% 0%,100% 100%;}}
  .stApp {
    background-image: radial-gradient(900px circle at 18% 18%, rgba(167,139,250,0.06) 0%, transparent 55%),
                      radial-gradient(1000px circle at 82% 78%, rgba(255,154,98,0.06) 0%, transparent 55%),
                      linear-gradient(180deg,#fbf9ff,#ffffff);
    background-repeat: no-repeat;
    animation: globalFloatGrad 22s ease-in-out infinite;
  }
</style>
"""

# App theme CSS - copied from frontend_reference.py
APP_THEME_CSS = """
<style>
  .stButton>button {background:#f7e8ff; color:#2c3e50; border:1px solid #e6dafe; border-radius:12px; padding:0.5rem 1.2rem; box-shadow:0 6px 18px rgba(167,139,250,0.12);} 
  .stButton>button:hover {filter:brightness(1.05);} 
  .stTextInput>div>div>input, .stTextArea>div>div>textarea {border:1.5px solid #e8eaf2; border-radius:10px; background:#ffffff; color:#2c3e50;}
  .stTextInput>div>div>input::placeholder, .stTextArea>div>div>textarea::placeholder {color:#9aa1a9;}
  .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {border-color:#d6d0f0; box-shadow:0 0 0 3px rgba(167,139,250,0.12);} 
  [data-testid="stSidebarContent"] {
    background:#ffffff !important;
    background-image:none !important;
    backdrop-filter:none !important;
  }
  [data-testid="stFileUploaderDropzone"] {background: radial-gradient(800px circle at 20% 20%, rgba(167,139,250,0.10) 0%, transparent 55%), radial-gradient(900px circle at 80% 75%, rgba(255,154,98,0.10) 0%, transparent 55%), #ffffff; border:1.5px dashed #d6d0f0;}
  .login-fixed {width:640px; margin:0 auto;}
  .upload-section .stButton>button {width:100%; background:linear-gradient(135deg,#a78bfa 0%, #ff7e5f 60%, #feb47b 100%); color:#ffffff; border:1px solid #e6dafe; box-shadow:0 10px 24px rgba(167,139,250,0.16); margin-top:14px;} 
  .logout-top .stButton>button {white-space: nowrap; min-width: 200px; padding:10px 18px;}
  [data-testid="stSidebarContent"] .stButton {margin-top:14px;}
  [data-testid="stSidebarContent"] .stButton>button {width:100%; background:#ffffff; color:#2c3e50; border:1px solid #e6dafe; box-shadow:0 6px 18px rgba(167,139,250,0.12);} 
  [data-testid="stSidebarContent"] .stButton>button:hover {filter:brightness(1.03); box-shadow:0 8px 22px rgba(167,139,250,0.16);} 
  [data-baseweb="notification"] {
    background: linear-gradient(180deg,#f5f3ff 0%, #ffffff 100%) !important;
    border:1px solid #c4b5fd !important;
    border-left:4px solid #a78bfa !important;
    color:#1f2937 !important;
    box-shadow:0 8px 22px rgba(167,139,250,0.16) !important;
  }
  [data-testid="stAlertContentWarning"], [data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"] {color:#1f2937 !important;}
  .profile-card {display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; padding:14px; border-radius:14px; background:#ffffff; border:1px solid #e6dafe; box-shadow:0 6px 18px rgba(167,139,250,0.10);} 
  .profile-avatar {width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#a78bfa 0%, #feb47b 60%, #ff7e5f 100%); color:#fff; font-weight:700; font-size:24px; box-shadow:0 8px 20px rgba(167,139,250,0.20);} 
  .profile-name {font-weight:800; color:#2c3e50; font-size:1.05rem;}
  .profile-role {color:#6b7280; font-size:.95rem;}
</style>
"""

LOGIN_BG_CSS = """
<style>
  .login-hero-layer {position:fixed; inset:0; z-index:0; pointer-events:none;
    background-image: radial-gradient(1100px circle at 20% 20%, rgba(167,139,250,0.12) 0%, transparent 55%), radial-gradient(1100px circle at 80% 75%, rgba(255,154,98,0.12) 0%, transparent 55%), linear-gradient(180deg,#faf7ff,#ffffff);
    background-repeat:no-repeat;
    animation: floatGrad 16s ease-in-out infinite;
  }
  @keyframes floatGrad {0%{background-position:0% 0%,100% 100%;}50%{background-position:20% 10%,80% 90%;}100%{background-position:0% 0%,100% 100%;}}
  @keyframes flowLight {0%{transform:translate3d(-16%, -12%, 0) rotate(0deg); filter:hue-rotate(0deg);}50%{transform:translate3d(18%, 12%, 0) rotate(180deg); filter:hue-rotate(18deg);}100%{transform:translate3d(-16%, -12%, 0) rotate(360deg); filter:hue-rotate(0deg);}}
  .login-hero-layer::before {content:""; position:absolute; inset:-10%; pointer-events:none;
    background:
      radial-gradient(580px circle at 26% 24%, rgba(255,154,98,0.16) 0%, transparent 62%),
      conic-gradient(from 0deg at 50% 50%, rgba(167,139,250,0.12), rgba(255,154,98,0.14), rgba(255,255,255,0.0), rgba(167,139,250,0.12));
    filter: blur(26px);
    animation: flowLight 16s ease-in-out infinite;
    opacity: 0.85;
  }
  .login-hero-layer::after {content:""; position:absolute; inset:-12%; pointer-events:none;
    background:
      radial-gradient(520px circle at 74% 72%, rgba(167,139,250,0.12) 0%, transparent 62%),
      conic-gradient(from 120deg at 50% 50%, rgba(255,154,98,0.10), rgba(167,139,250,0.12), rgba(255,255,255,0.0), rgba(255,154,98,0.10));
    filter: blur(22px);
    animation: flowLight 20s ease-in-out infinite reverse;
    opacity: 0.80;
  }
  .login-content {position:relative; z-index:1;}
</style>
"""

QA_THEME_CSS = """
<style>
  div[data-testid="stChatMessage"] {
    margin:12px 0 !important;
    display:grid !important;
    grid-template-columns:40px 1fr !important;
    gap:12px !important;
    align-items:start !important;
    width:100% !important;
    overflow:visible !important;
  }
  /* 头像容器（第一个子元素） */
  div[data-testid="stChatMessage"] > div:first-child {
    width:40px !important; height:40px !important;
    border-radius:50% !important; display:flex !important; align-items:center !important; justify-content:center !important;
    background:#e6eaf5 !important; color:#1f2937 !important; font-weight:700 !important; font-size:20px !important;
    box-shadow:0 2px 8px rgba(0,0,0,0.12) !important;
    visibility:visible !important; opacity:1 !important;
  }
  /* 问答内容容器 */
  div[data-testid="stChatMessage"] > div[data-testid="stChatMessageContent"] {
    border:1.5px solid #e6dafe !important; border-radius:12px !important; padding:12px 14px !important;
    background:#ffffff !important; background-color:#ffffff !important; box-shadow:0 6px 18px rgba(167,139,250,0.08) !important;
    min-width:0 !important; overflow:visible !important;
  }
  /* 通过 aria-label 精确匹配角色（直接在内容容器上） */
  div[data-testid="stChatMessage"] > div[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
    background:#f5f8ff !important;
    background-color:#f5f8ff !important;
    border-color:#a3c5ff !important;
    box-shadow:0 6px 18px rgba(134,179,255,0.16) !important;
    border-radius:12px !important;
    border-left:4px solid #6aa0ff !important;
    width:100% !important;
    box-sizing:border-box !important;
  }
  .e1ypd8m70[data-testid="stChatMessage"] { display:grid !important; grid-template-columns:40px 1fr !important; gap:12px !important; align-items:start !important; }
  div[data-testid="stChatMessage"] > div:first-child { grid-column:1 !important; grid-row:1 !important; justify-self:center !important; align-self:start !important; }
  div[data-testid="stChatMessage"] > div[data-testid="stChatMessageContent"] { grid-column:2 !important; grid-row:1 !important; width:100% !important; box-sizing:border-box !important; }
  div[data-testid="stChatMessageContent"][aria-label="Chat message from user"] .stMarkdown, 
  div[data-testid="stChatMessageContent"][aria-label="Chat message from user"] .stMarkdown p { text-align:left !important; }
  /* 移除 Markdown 层的背景，避免仅有灰线的块被着色 */
  div[data-testid="stChatMessage"].st-emotion-cache-1fee4w7.e1ypd8m70 {
    background:#f5f8ff !important;
    border:1.5px solid #a3c5ff !important;
    border-radius:12px !important;
    padding:8px !important;
    box-shadow:0 6px 18px rgba(134,179,255,0.16) !important;
  }
  div.st-emotion-cache-1dgsum0.e1ypd8m72 {
    width:40px !important;
    height:40px !important;
    border-radius:50% !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    background:#e6eaf5 !important;
    color:#1f2937 !important;
    font-weight:700 !important;
    font-size:20px !important;
    line-height:40px !important;
    flex:0 0 40px !important;
    position:relative !important;
  }
  /* Reference Sources 半透明背景，仅作用于聊天内容内的展开区 */
  div[data-testid="stChatMessageContent"] div[data-testid="stExpanderDetails"] {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid #e6dafe !important;
    border-radius: 10px !important;
    padding: 8px 10px !important;
  }
  /* 强化选择器权重，仅为用户消息内容着色 */
  div[data-testid="stChatMessage"].st-emotion-cache-1fee4w7.e1ypd8m70 > div.st-emotion-cache-1flajlm.e1ypd8m71[aria-label="Chat message from user"] {
    background:#f5f8ff !important;
    border:1.5px solid #a3c5ff !important;
    border-left:4px solid #6aa0ff !important;
    border-radius:12px !important;
    box-shadow:0 6px 18px rgba(134,179,255,0.16) !important;
  }
</style>
"""

APP_BG_CSS = """
<style>
  .app-hero-layer {position:fixed; inset:0; z-index:0; pointer-events:none;
    background-image: radial-gradient(1100px circle at 20% 20%, rgba(167,139,250,0.12) 0%, transparent 55%), radial-gradient(1100px circle at 80% 75%, rgba(134,179,255,0.12) 0%, transparent 55%), linear-gradient(180deg,#faf7ff,#ffffff);
    background-repeat:no-repeat;
    animation: appFloatGrad 16s ease-in-out infinite;
  }
  @keyframes appFloatGrad {0%{background-position:0% 0%,100% 100%;}50%{background-position:20% 10%,80% 90%;}100%{background-position:0% 0%,100% 100%;}}
  @keyframes appFlowLight {0%{transform:translate3d(-14%, -10%, 0) rotate(0deg);}50%{transform:translate3d(16%, 10%, 0) rotate(180deg);}100%{transform:translate3d(-14%, -10%, 0) rotate(360deg);}}
  .app-hero-layer::before {content:""; position:absolute; inset:-10%; pointer-events:none;
    background:
      radial-gradient(560px circle at 26% 24%, rgba(134,179,255,0.14) 0%, transparent 62%),
      conic-gradient(from 0deg at 50% 50%, rgba(167,139,250,0.12), rgba(134,179,255,0.14), rgba(255,255,255,0.0), rgba(167,139,250,0.12));
    filter: blur(24px);
    animation: appFlowLight 16s ease-in-out infinite;
    opacity: 0.85;
  }
  .app-hero-layer::after {content:""; position:absolute; inset:-12%; pointer-events:none;
    background:
      radial-gradient(500px circle at 74% 72%, rgba(167,139,250,0.12) 0%, transparent 62%),
      conic-gradient(from 120deg at 50% 50%, rgba(134,179,255,0.10), rgba(167,139,250,0.12), rgba(255,255,255,0.0), rgba(134,179,255,0.10));
    filter: blur(20px);
    animation: appFlowLight 20s ease-in-out infinite reverse;
    opacity: 0.80;
  }
  .app-content-wrap {position:relative; z-index:1;}
</style>
"""

IDENTITY_CSS = """
<style>
  .identity-section {position:relative;z-index:1;padding:0.25rem 0;min-height:auto;display:flex;align-items:center;justify-content:center;}
  .identity-container {width:900px;max-width:900px;margin:0 auto;padding:0 8px;}
  .identity-title {text-align:center;font-size:clamp(2.2rem,4.2vw,3.2rem);font-weight:800;color:#2c3e50;margin:0;}
  .identity-subtitle {text-align:center;font-size:1.05rem;color:#6b7280;margin:0.35rem 0 0.5rem;}
  .identity-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:stretch;}
  .id-card-block {position:relative; min-height:360px; height:100%;}
  .id-card {position:relative;background:#ffffff; border-radius:16px;border:2px solid #e6dafe;box-shadow:0 8px 22px rgba(167,139,250,0.16);padding:20px;transition:box-shadow .3s,border-color .3s,transform .3s; height:100%; padding-bottom:64px; display:flex; flex-direction:column;}
  .id-card:hover {border-color:#d6d0f0;box-shadow:0 12px 26px rgba(167,139,250,0.22);transform:translateY(-2px);} 
  .id-icon {width:64px;height:64px;border-radius:16px;background:transparent;display:flex;align-items:center;justify-content:center;color:#ff7e5f;font-size:30px;border:2px solid #ff7e5f;box-shadow:0 6px 16px rgba(255,126,95,0.18);} 
  .id-icon {border:none; box-shadow:none;}
  .id-h2 {font-size:1.6rem;font-weight:800;color:#2c3e50;margin:14px 0 8px 0;} 
  .id-desc {color:#6b7280;line-height:1.6;margin-bottom:10px;} 
  .id-divider {height:1px;background:#e5e7eb;margin:12px 0;} 
  .id-list-title {font-size:.9rem;font-weight:700;color:#2c3e50;letter-spacing:.06em;margin-bottom:8px;} 
  .id-list {list-style:none;padding:0;margin:0;display:grid;gap:8px;} 
  .id-list li {display:flex;align-items:center;gap:8px;color:#6b7280;font-size:.95rem;} 
  .dot {width:6px;height:6px;border-radius:50%;background:linear-gradient(135deg,#ff7e5f,#a78bfa);} 
  .identity-footer {text-align:center;color:#9aa1a9;font-size:.9rem;margin-top:8px;}
  .id-cta-wrap {position:absolute; left:20px; right:20px; bottom:20px;}
  .id-cta-wrap .stButton>button {display:inline-flex;align-items:center;justify-content:center;width:100%;padding:12px 16px;border-radius:14px;background:linear-gradient(135deg,#ff6a4a 0%, #ff7e5f 50%, #ff4d4f 100%);color:#fff;text-decoration:none;font-weight:700;box-shadow:0 10px 26px rgba(255,77,79,0.28); border:none;} 
  .id-cta-wrap .stButton>button:hover {filter:brightness(1.06);} 
</style>
"""

def safe_markdown(text):
    """Safely escape $ signs before Markdown rendering to prevent LaTeX triggering."""
    
    import re
    # Replace all $ with \$ to prevent LaTeX rendering
    # This handles both single $ and $$ patterns
    text = text.replace('$', r'\$')
    # Render with markdown (without unsafe_allow_html)
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
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'role_selected' not in st.session_state:
            st.session_state.role_selected = False
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'current_file_id' not in st.session_state:
            st.session_state.current_file_id = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'pending_question' not in st.session_state:
            st.session_state.pending_question = None
        if 'page' not in st.session_state:
            st.session_state.page = 'marketing'
    
    def login_page(self):
        """Login page"""
        st.markdown(GLOBAL_CSS + APP_THEME_CSS + LOGIN_BG_CSS, unsafe_allow_html=True)
        st.markdown("<div class='login-hero-layer'></div><div class='login-content'><div class='login-fixed'>", unsafe_allow_html=True)
        st.title("Login")
        
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
            st.markdown("<a name='register'></a>", unsafe_allow_html=True)
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
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#9aa1a9;font-size:.95rem;margin-top:8px;'>© 2025 RentalPeace. All rights reserved.</div>", unsafe_allow_html=True)
    
    def guidance_page(self):
        st.set_page_config(page_title="How to Use", page_icon="📘", layout="centered")
        st.markdown(GLOBAL_CSS + APP_THEME_CSS, unsafe_allow_html=True)
        st.markdown(
            """
            <div style="max-width:860px;margin:0 auto;padding:24px 16px;">
              <h1 style="text-align:center;color:#2c3e50;margin-bottom:10px;">How to Use Rental Peace</h1>
              <p style="text-align:center;color:#6b7280;">Welcome to Rental Peace — your AI-powered assistant for understanding rental agreements with clarity and confidence.</p>
              <hr/>
              <h3 style="color:#2c3e50;">Step 1 · Sign in</h3>
              <p style="color:#6b7280;">Log in to your Rental Peace account to access the AI interpretation tools. If you’re new here, you can sign up on the same page.</p>
              <hr/>
              <h3 style="color:#2c3e50;">Step 2 · Choose Your Role</h3>
              <p style="color:#6b7280;">After logging in, select your identity: I’m a Tenant or I’m a Landlord. Both options lead to the next step.</p>
              <hr/>
              <h3 style="color:#2c3e50;">Step 3 · Upload Your Contract</h3>
              <p style="color:#6b7280;">Click the Browse File or Upload button to submit your rental agreement. Your document stays private and is used only for AI analysis.</p>
              <hr/>
              <h3 style="color:#2c3e50;">Step 4 · Let AI Interpret</h3>
              <p style="color:#6b7280;">AI analyzes your contract and provides a clear summary of key clauses, highlights important rights or risks, and explains complex terms in plain language.</p>
              <hr/>
              <h3 style="color:#2c3e50;">Step 5 · Ask Follow-up Questions</h3>
              <p style="color:#6b7280;">On the Q&A page, ask anything such as: What does this clause mean? Can I terminate early? Is this rent increase legal?</p>
              <hr/>
              <h3 style="color:#2c3e50;">Step 6 · Save or Share Insights</h3>
              <p style="color:#6b7280;">Copy or download the AI interpretation results for future reference or share them with your agent, lawyer, or roommate.</p>
              <hr/>
              <p style="text-align:center;color:#2c3e50;">Rental Peace makes legal language understandable — sign with confidence and live with peace of mind.</p>
              <div style='text-align:center;color:#9aa1a9;font-size:.95rem;margin-top:8px;'>© 2025 RentalPeace. All rights reserved.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    def role_selection_page(self):
        st.set_page_config(page_title="Select Role", page_icon="👤", layout="centered")
        st.markdown(GLOBAL_CSS + APP_THEME_CSS, unsafe_allow_html=True)
        st.markdown(IDENTITY_CSS, unsafe_allow_html=True)
        # 顶部左侧固定 Logout 按钮（不占用卡片区域）
        top_left, top_right = st.columns([2, 7])
        with top_left:
            st.markdown("<div class='logout-top'>", unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <section class="identity-section">
              <div class="identity-container">
                <div class="identity-title">I am a...</div>
                <div class="identity-subtitle">Select your role to get personalized insights tailored to your needs</div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="id-card-block">
                  <div class="id-card">
                  <div class="id-icon">👤</div>
                  <div class="id-h2">I'm a Tenant</div>
                  <div class="id-desc">Understand your rights, obligations, and key terms in your rental agreement</div>
                  <div class="id-divider"></div>
                  <div class="id-list-title">WHAT YOU'LL GET:</div>
                  <ul class="id-list">
                    <li><span class="dot"></span>Know your rental rights</li>
                    <li><span class="dot"></span>Understand payment terms</li>
                    <li><span class="dot"></span>Check notice periods</li>
                    <li><span class="dot"></span>Review maintenance clauses</li>
                  </ul>
                  </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='id-cta-wrap'>", unsafe_allow_html=True)
            if st.button("Continue as Tenant →", use_container_width=True):
                result = self.user_manager.set_user_role(st.session_state.user_id, 'tenant')
                if result["success"]:
                    st.session_state.user_role = 'tenant'
                    st.session_state.role_selected = True
                    st.rerun()
                else:
                    st.error("Failed to set role, please try again")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                """
                <div class="id-card-block">
                  <div class="id-card">
                  <div class="id-icon">🏢</div>
                  <div class="id-h2">I'm a Landlord</div>
                  <div class="id-desc">Review tenant agreements and protect your property interests</div>
                  <div class="id-divider"></div>
                  <div class="id-list-title">WHAT YOU'LL GET:</div>
                  <ul class="id-list">
                    <li><span class="dot"></span>Verify contract completeness</li>
                    <li><span class="dot"></span>Check legal compliance</li>
                    <li><span class="dot"></span>Review tenant obligations</li>
                    <li><span class="dot"></span>Understand liability terms</li>
                  </ul>
                  </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div class='id-cta-wrap'>", unsafe_allow_html=True)
            if st.button("Continue as Landlord →", use_container_width=True):
                result = self.user_manager.set_user_role(st.session_state.user_id, 'landlord')
                if result["success"]:
                    st.session_state.user_role = 'landlord'
                    st.session_state.role_selected = True
                    st.rerun()
                else:
                    st.error("Failed to set role, please try again")
            st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="identity-footer">Don't worry, you can always change this later in your settings</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='text-align:center;color:#9aa1a9;font-size:.95rem;margin-top:8px;'>© 2025 RentalPeace. All rights reserved.</div>", unsafe_allow_html=True)
    
    def marketing_page(self):
        """Marketing page - styled like frontend_reference.py"""
        st.markdown(HERO_CSS, unsafe_allow_html=True)
        st.markdown("""
            <div class="hero-layer"></div>
            <section class="hero-section">
              <div class="hero-container">
                <div class="marketing-hero">
                  <div>
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <div class="social-avatars">
                      <span class="avatar"></span>
                      <span class="avatar"></span>
                      <span class="avatar"></span>
                      <span class="avatar"></span>
                    </div>
                    <div class="rating">
                      <span>⭐</span>
                      <span style="font-weight:600;color:#2c3e50;">4.9</span>
                      <span>• 2,500+ happy users in Singapore</span>
                    </div>
                  </div>
                  <h1 class="hero-title">Understand Your<br><span class="gradient-primary">Rental Agreement</span><br>in Minutes</h1>
                  <p class="hero-desc">AI-powered contract analysis for tenants and landlords. Get instant insights, ask questions, and extract key information from any rental agreement.</p>
                  <div class="cta-row">
                    <a class="cta cta-primary" href="?page=login" target="_self">Get Started Free →</a>
                    <a class="cta cta-outline" href="?page=guidance" target="_self">How to Use</a>
                  </div>
                  <div class="trust-line"></div>
                  <div class="trust">
                    <span> No credit card required</span>
                    <span> 100% secure & private</span>
                  </div>
                  <div class="copyright">© 2025 RentalPeace. All rights reserved.</div>
                </div>
              </div>
            </section>
        """, unsafe_allow_html=True)
    
    def init_user_rag_system(self):
        """Initialize user's RAG system"""
        if st.session_state.rag_system is None:
            st.session_state.rag_system = AdvancedContractRAG(                
                api_key = st.secrets["OPENAI_API_KEY"],
                model=st.secrets.get("OPENAI_MODEL", "gpt-4o")
            )
            # Set user-specific cache directory
            user_cache_dir = Path(f"user_data/{st.session_state.user_id}/cache")
            user_cache_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.rag_system.cache_dir = user_cache_dir
    
    def main_app(self):
        """Main application interface"""
        st.set_page_config(page_title="Contract Assistant", page_icon="📄", layout="wide")
        st.markdown(GLOBAL_CSS + APP_THEME_CSS + APP_BG_CSS, unsafe_allow_html=True)
        st.markdown("<div class='app-hero-layer'></div><div class='app-content-wrap'>", unsafe_allow_html=True)
        
        # Initialize RAG system
        self.init_user_rag_system()
        
        # Sidebar
        with st.sidebar:
            username_display = st.session_state.username or "Guest"
            role_display = "🏡 Tenant" if st.session_state.get('user_role') == 'tenant' else ("🏢 Landlord" if st.session_state.get('user_role') == 'landlord' else "Unknown")
            avatar_text = (username_display[:1].upper() if isinstance(username_display, str) and len(username_display) > 0 else "?")
            st.markdown(f"<div class='profile-card'><div class='profile-avatar'>{avatar_text}</div><div class='profile-name'>{username_display}</div><div class='profile-role'>{role_display}</div></div>", unsafe_allow_html=True)
            
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
        st.title("📄 RentalPeace-Your Intelligent Assistant")
        
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
            st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Tab2: Q&A
        with tab2:
            if not st.session_state.current_file_id:
                st.warning("Please upload or load a file first")
            else:
                # Inject CSS for quick questions and Q&A theme
                st.markdown(QUICK_QUESTION_CSS + QA_THEME_CSS, unsafe_allow_html=True)
                
                # ⭐ New: Display current contract information in use
                if current_file_info:
                    st.info(f"🎯 Current Q&A for contract: **{current_file_info['filename']}**")
                
                # 快捷提问气泡 - 始终显示，使用expander
                with st.expander("💡 Quick Questions - Click to expand", expanded=not bool(st.session_state.messages)):
                    st.caption("Click a question below to get instant answers:")
                    
                    # 根据用户角色显示不同的问题
                    if st.session_state.user_role == 'tenant':
                        # 租客问题
                        quick_questions = [
                            "What is the monthly rent amount?",
                            "What is the lease duration?",
                            "What is the security deposit amount?",
                            "When is the rent due each month?",
                            "What are my maintenance responsibilities?",
                            "What is the pet policy?",
                            "What are the termination conditions?",
                            "Who pays for utilities?"
                        ]
                        question_icons = ["💰", "📅", "🏦", "📆", "🔧", "🐕", "🚪", "💡"]
                        question_labels = [
                            "Monthly Rent",
                            "Lease Duration",
                            "Security Deposit",
                            "Payment Due",
                            "Maintenance",
                            "Pet Policy",
                            "Termination",
                            "Utilities"
                        ]
                    else:  # landlord
                        # 房东问题
                        quick_questions = [
                            "What are the tenant's payment obligations?",
                            "What are the late payment penalties?",
                            "What are my maintenance obligations as landlord?",
                            "What are the property access rights?",
                            "What are the lease renewal terms?",
                            "What are the tenant's restrictions?",
                            "What are the eviction conditions?",
                            "What are my liability protections?"
                        ]
                        question_icons = ["💵", "⚠️", "🏗️", "🔑", "🔄", "⛔", "📋", "🛡️"]
                        question_labels = [
                            "Payment Terms",
                            "Late Penalties",
                            "Maintenance",
                            "Access Rights",
                            "Renewal Terms",
                            "Restrictions",
                            "Eviction",
                            "Liability"
                        ]
                    
                    # 使用列布局显示问题按钮
                    cols = st.columns(4)
                    for idx, (icon, label, question) in enumerate(zip(question_icons, question_labels, quick_questions)):
                        col_idx = idx % 4
                        with cols[col_idx]:
                            # 创建按钮标签：emoji + 简短文字
                            button_label = f"{icon} {label}"
                            if st.button(button_label, key=f"quick_q_{idx}", use_container_width=True, help=question):
                                # 模拟用户点击，设置问题
                                st.session_state.pending_question = question
                                st.rerun()
                    
                    # 显示问题文本（用于用户查看）
                    with st.expander("📝 View all quick questions", expanded=False):
                        for icon, label, question in zip(question_icons, question_labels, quick_questions):
                            st.markdown(f"**{icon} {label}**: {question}")
                    
                    st.divider()
                
                # 处理待处理的问题（从快捷按钮点击）- 优化：立即处理，无需额外rerun
                if 'pending_question' in st.session_state and st.session_state.pending_question:
                    prompt = st.session_state.pending_question
                    st.session_state.pending_question = None  # 清除待处理问题
                    
                    # 添加用户问题到历史
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # 获取AI回答
                    with st.spinner("🤔 Thinking..."):
                        response = st.session_state.rag_system.ask_question(prompt)
                        
                        # 保存到历史
                        self.cache_manager.save_qa_history(
                            st.session_state.user_id,
                            st.session_state.current_file_id,
                            prompt,
                            response["answer"],
                            response.get("sources", [])
                        )
                        
                        # 保存助手回答到历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response.get("sources", [])
                        })
                    
                    st.rerun()  # 重新加载以显示对话
                
                
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
                    with st.chat_message(message["role"], avatar=("👤" if message["role"] == "user" else "🤖")):
                        # 转义$符号以防止LaTeX渲染
                        content = message["content"].replace("$", "\\$")
                        st.markdown(content)
                        # Display sources if available (same format as new messages)
                        if message.get("sources"):
                            with st.expander("📚 Reference Sources"):
                                for i, source in enumerate(message["sources"], 1):
                                    page_number = source.get('page', 'N/A')
                                    if page_number is not None and isinstance(page_number, int):
                                        page_number += 1  # Change page numbering from 0 to 1-based
                                    else:
                                        page_number = 'N/A'
                                    
                                    # 显示相似度分数
                                    # similarity_score = source.get('similarity_score', 0)
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
                        st.error(f"❌ Document validation failed: {e}")
                        st.stop()
                    
                    # Display user question immediately
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user", avatar="👤"):
                        st.write(prompt)
                    
                    # Display assistant thinking
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("🤔 Thinking..."):
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
                                        page_number = source.get('page', 'N/A')
                                        if page_number is not None and isinstance(page_number, int):
                                            page_number += 1  # Change page numbering from 0 to 1-based
                                        else:
                                            page_number = 'N/A'
                                        
                                        # 显示相似度分数
                                        # similarity_score = source.get('similarity_score', 0)
                                        st.markdown(f"**📄 Source {i} - Page {page_number}**")
                                        
                                        content = source.get('content', '')
                                        
                                        # Display preview (first 500 characters)
                                        preview_length = 500
                                        if len(content) <= preview_length:
                                            st.text_area(
                                                f"Source content_{i}",
                                                content,
                                                height=150,
                                                key=f"new_source_preview_{len(st.session_state.messages)}_{i}",  # ← Add message count
                                                label_visibility="collapsed"
                                            )
                                        else:
                                            # Display preview
                                            st.text_area(
                                                f"Source content preview_{i}",
                                                content[:preview_length] + "...",
                                                height=150,
                                                key=f"new_source_preview_long_{len(st.session_state.messages)}_{i}",  # ← Unique key
                                                label_visibility="collapsed"
                                            )
                                            # Provide option to view full content
                                            with st.expander(f"🔍 View full content ({len(content)} Characters)"):
                                                st.text_area(
                                                        f"Full content_{i}",
                                                        content,
                                                        height=300,
                                                        key=f"new_source_full_{len(st.session_state.messages)}_{i}",  # ← Unique key
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
                    ["brief", "comprehensive", "key points"],
                    format_func=lambda x: x.title()  # 首字母大写显示
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
                            
                            safe_markdown(summary)

        
        # Tab4: Extract Info
        with tab4:
            if not st.session_state.current_file_id:
                st.warning("Please upload or load a file first")
            else:
                if st.button("Extract Key Information"):
                    # Check cache
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
                        {"No.": idx + 1, "Keyword": k, "Details": v} for idx, (k, v) in enumerate(key_info.items())
                    ])
                    st.dataframe(
                        df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "No.": st.column_config.NumberColumn(
                                "No.",
                                width=50  # 自定义宽度，更小
                            ),
                            "Keyword": st.column_config.TextColumn(
                                "Keyword",
                                width="medium"
                            ),
                            "Details": st.column_config.TextColumn(
                                "Details",
                                width="large"
                            )
                        }
                    )
        
        # Tab5: Compare (simplified version)
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
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#9aa1a9;font-size:.95rem;margin-top:8px;'>© 2025 RentalPeace. All rights reserved.</div>", unsafe_allow_html=True)
    
    def run(self):
        """Run application"""
        st.set_page_config(page_title="Contract Assistant", page_icon="📄", layout="wide")
        
        # Handle page routing from URL parameters
        params = dict(st.query_params)
        if 'page' in params:
            page_val = params.get('page', 'marketing')
            if isinstance(page_val, list):
                page_val = page_val[0]
            st.session_state.page = page_val
        
        if st.session_state.authenticated and not st.session_state.get('role_selected', False):
            existing_role = self.user_manager.get_user_role(st.session_state.user_id)
            if existing_role:
                st.session_state.user_role = existing_role
                st.session_state.role_selected = True
                st.rerun()
            else:
                self.role_selection_page()
        elif st.session_state.authenticated:
            self.main_app()
        else:
            if st.session_state.page == 'login':
                self.login_page()
            elif st.session_state.page == 'identity':
                self.role_selection_page()
            elif st.session_state.page == 'guidance':
                self.guidance_page()
            else:
                self.marketing_page()
