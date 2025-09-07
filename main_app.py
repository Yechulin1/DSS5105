import streamlit as st
import os
from typing import List, Dict
import json
from datetime import datetime

# Import our RAG system
from rag_engine import RAGSystem

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

def initialize_rag_system():
    """Initialize the RAG system with contract documents"""
    if st.session_state.rag_system is None:
        st.session_state.rag_system = RAGSystem()

def load_contract_files():
    """Load contract files from various locations"""
    # Find all markdown and text files in project directory
    contract_files = []
    
    # Check root directory
    try:
        for file in os.listdir("."):
            if file.endswith((".md", ".txt", ".pdf")):
                contract_files.append(file)
    except:
        pass
    
    # Check documents folder if it exists
    if os.path.exists("documents"):
        try:
            for file in os.listdir("documents"):
                if file.endswith((".md", ".txt", ".pdf")):
                    contract_files.append(f"documents/{file}")
        except:
            pass
    
    return contract_files

def main():
    # Page config
    st.set_page_config(
        page_title="🏠 Tenant AI Assistant",
        page_icon="🏠",
        layout="wide"
    )
    
    # Header
    st.title("🏠 Real Estate AI Assistant")
    st.markdown("### Your Smart Tenant Support Chatbot")
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Info")
        
        # Contract file selector
        st.subheader("📄 Contract Management")
        available_files = load_contract_files()
        
        if available_files:
            selected_file = st.selectbox(
                "Select Contract File:",
                available_files,
                help="Choose which contract to load"
            )
            
            if st.button("📥 Load Contract"):
                with st.spinner(f"Loading {selected_file}..."):
                    if st.session_state.rag_system.load_document(selected_file):
                        st.success(f"✅ Loaded: {selected_file}")
                    else:
                        st.error(f"❌ Failed to load: {selected_file}")
            
            # File upload option
            st.divider()
            uploaded_file = st.file_uploader(
                "Upload New Contract",
                type=['txt', 'md', 'pdf'],
                help="Upload your contract file"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                file_path = f"{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the uploaded file
                if st.session_state.rag_system.load_document(file_path):
                    st.success(f"✅ Uploaded and loaded: {uploaded_file.name}")
                else:
                    st.error(f"❌ Failed to process: {uploaded_file.name}")
        else:
            st.warning("No contract files found. Please upload one.")
            
            # File upload option
            uploaded_file = st.file_uploader(
                "Upload Contract",
                type=['txt', 'md', 'pdf'],
                help="Upload your contract file"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                file_path = f"{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the uploaded file
                if st.session_state.rag_system.load_document(file_path):
                    st.success(f"✅ Uploaded and loaded: {uploaded_file.name}")
                    st.rerun()
        
        st.divider()
        st.info("""
        **Features:**
        - Contract Q&A
        - Rent & Payment Info
        - Maintenance Guidelines
        - House Rules
        
        **Powered by:**
        - RAG Technology
        - OpenAI GPT-3.5
        """)
        
        # API Key input
        st.divider()
        api_key = st.text_input("OpenAI API Key:", type="password", 
                               help="Enter your OpenAI API key to enable AI responses")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        # Sample questions
        st.divider()
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "When is my rent due?",
            "Can I keep a pet?",
            "What's the penalty for late payment?",
            "Who handles aircon servicing?",
            "Can I terminate early?",
            "What are the quiet hours?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.messages.append({"role": "user", "content": q})
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        st.text(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your tenancy agreement..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            if st.session_state.rag_system and api_key:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, sources = st.session_state.rag_system.answer_question(prompt)
                        st.write(response)
                        
                        # Show sources
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.text(f"• {source}")
                        
                        # Save assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": "\n".join(sources) if sources else ""
                        })
            elif not api_key:
                with st.chat_message("assistant"):
                    # Use simple response without API key
                    with st.spinner("Thinking..."):
                        response, sources = st.session_state.rag_system.answer_question(prompt)
                        st.write(response)
                        
                        # Show sources
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.text(f"• {source}")
                        
                        # Save assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": "\n".join(sources) if sources else ""
                        })
            else:
                with st.chat_message("assistant"):
                    st.error("❌ Please load a contract first using the sidebar.")
    
    with col2:
        st.subheader("📊 System Status")
        
        # API Status Card
        with st.container():
            if api_key and 'api_status' in st.session_state:
                if st.session_state.api_status == "connected":
                    st.success("🟢 **AI Mode Active**")
                    st.caption("Using GPT-3.5 for intelligent responses")
                else:
                    st.warning("🟡 **Basic Mode**")
                    st.caption("Using pattern matching (API key invalid)")
            elif api_key:
                st.info("🔵 **API Key Entered**")
                st.caption("Click 'Test' in sidebar to verify")
            else:
                st.warning("🟡 **Basic Mode**")
                st.caption("Add API key for better responses")
        
        st.divider()
        
        if st.session_state.rag_system and st.session_state.rag_system.chunks:
            # Display some stats about loaded contract
            st.metric("Documents Loaded", len(st.session_state.rag_system.documents))
            st.metric("Text Chunks", len(st.session_state.rag_system.chunks))
            
            # Show loaded files
            if st.session_state.rag_system.documents:
                st.divider()
                st.caption("📄 Loaded Files:")
                for doc in st.session_state.rag_system.documents:
                    st.caption(f"• {doc['filename']}")
            
            st.divider()
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("📅 Check Payment Schedule"):
                question = "What is my payment schedule and when is rent due?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
                
            if st.button("🔧 Report Maintenance"):
                question = "What are the maintenance responsibilities and how do I report issues?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
                
            if st.button("📜 View House Rules"):
                question = "What are all the house rules I need to follow?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
                
            if st.button("🚪 Termination Info"):
                question = "What are the terms for early termination?"
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        else:
            st.info("📄 Please load a contract using the sidebar")
    
    # Footer
    st.divider()
    st.caption("🤖 Powered by RAG Technology | DSS5105 Capstone Project")

if __name__ == "__main__":
    main()