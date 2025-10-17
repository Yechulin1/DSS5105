# advanced_rag_system.py
"""
高级合同管理系统 - 使用LangChain实现
支持PDF解析、合同总结、智能问答等功能
"""

import os
from typing import List, Dict, Optional, Tuple
import hashlib
import pickle
from datetime import datetime

# LangChain核心组件
from langchain.document_loaders import PyMuPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import (
    RetrievalQA, 
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.callbacks import get_openai_callback

# 工具类
import numpy as np
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv()

class AdvancedContractRAG:
    """
    高级合同RAG系统
    特性：
    - 强大的PDF解析（支持复杂格式）
    - 智能文档分块
    - 语义向量搜索
    - 合同自动总结
    - 对话历史记忆
    - 多语言支持
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", language: str = "en"):
        """
        初始化高级RAG系统
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型 (gpt-3.5-turbo, gpt-4等)
            language: 语言设置 (en, zh等)
        """
        self.api_key = api_key
        self.model = model
        self.language = language
        
        # 初始化OpenAI组件
        self.llm = ChatOpenAI(
            temperature=0.01,
            model=model,
            openai_api_key=api_key,
            max_tokens=500
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key
        )
        
        # 文本分割器 - 智能分块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,        # 块大小
            chunk_overlap=200,     # 重叠部分保持上下文
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]  # 支持中英文
        )
        
        # 向量存储
        self.vectorstore = None
        self.retriever = None
        
        # 对话记忆
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",   # ✅ 告诉 memory：输入字段叫 question
            output_key="answer",
            return_messages=True,
            k=5  # 记住最近5轮对话
        )
        
        # 存储已加载的文档
        self.documents = {}
        self.contract_metadata = {}
        
        # 缓存目录
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_pdf(self, pdf_path: str, use_cache: bool = True) -> Dict:
        """
        加载并解析PDF文件
        
        Args:
            pdf_path: PDF文件路径
            use_cache: 是否使用缓存
            
        Returns:
            包含解析结果的字典
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"success": False, "error": f"File not found: {pdf_path}"}
        
        # 确保只有一个文档（新增）
        if self.ensure_single_document(str(pdf_path)):
            # 如果是同一个文件且已加载，直接返回
            return {
                "success": True, 
                "message": "Document already loaded",
                "stats": self.contract_metadata.get(str(pdf_path), {})
            }
    
        
        # 检查缓存
        cache_key = self._get_cache_key(pdf_path)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if use_cache and cache_path.exists():
            print(f"📂 Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.documents[str(pdf_path)] = cached_data['documents']
                self._rebuild_vectorstore()
                return {"success": True, "message": "Loaded from cache", "stats": cached_data['stats']}
        
        print(f"📄 Loading PDF: {pdf_path}")
        
        # 尝试多种PDF加载器
        documents = None
        loader_used = None
        
        # 方法1: PDFPlumber (最好的表格支持)
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            documents = loader.load()
            loader_used = "PDFPlumber"
            print(f"✅ Successfully loaded with PDFPlumber")
        except Exception as e:
            print(f"⚠️ PDFPlumber failed: {e}")
        
        # 方法2: PyMuPDF (最准确的文本提取)
        if documents is None or len(documents) == 0:
            try:
                loader = PyMuPDFLoader(str(pdf_path))
                documents = loader.load()
                loader_used = "PyMuPDF"
                print(f"✅ Successfully loaded with PyMuPDF")
            except Exception as e:
                print(f"⚠️ PyMuPDF failed: {e}")
        
        if documents is None or len(documents) == 0:
            return {"success": False, "error": "Failed to extract text from PDF"}
        
        # 提取元数据
        total_pages = len(documents)
        total_text = " ".join([doc.page_content for doc in documents])
        total_chars = len(total_text)
        
        # 智能文档分块
        split_documents = self.text_splitter.split_documents(documents)
        
        # 为每个块添加元数据
        for i, doc in enumerate(split_documents):
            doc.metadata.update({
                "source": str(pdf_path),
                "chunk_id": i,
                "loader": loader_used,
                "timestamp": datetime.now().isoformat()
            })
        
        # 存储文档
        self.documents[str(pdf_path)] = split_documents
        
        # 更新向量存储
        self._rebuild_vectorstore()
        
        # 统计信息
        stats = {
            "file": pdf_path.name,
            "pages": total_pages,
            "characters": total_chars,
            "chunks": len(split_documents),
            "loader": loader_used,
            "avg_chunk_size": total_chars // len(split_documents) if split_documents else 0
        }
        
        # 缓存处理结果
        if use_cache:
            cache_data = {
                "documents": split_documents,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"💾 Cached to: {cache_path}")
        
        # 存储合同元数据
        self.contract_metadata[str(pdf_path)] = stats
        
        return {"success": True, "message": f"Successfully loaded {pdf_path.name}", "stats": stats}
    
    def _get_cache_key(self, file_path: Path) -> str:
        """生成文件缓存键"""
        stat = file_path.stat()
        unique_str = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def _rebuild_vectorstore(self):
        """重建向量存储"""
        all_documents = []
        for docs in self.documents.values():
            all_documents.extend(docs)
        
        if all_documents:
            print(f"🔄 Building vector store with {len(all_documents)} chunks...")
            self.vectorstore = FAISS.from_documents(
                all_documents,
                self.embeddings
            )
            
            # 创建增强检索器
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 5,  # 返回5个最相关的块
                    "fetch_k": 10  # 先获取10个候选
                }
            )
            print(f"✅ Vector store ready")
    
    def summarize_contract(self, pdf_path: Optional[str] = None, 
                          summary_type: str = "comprehensive") -> str:
        """
        生成合同摘要
        
        Args:
            pdf_path: 指定PDF路径，None则总结所有已加载文档
            summary_type: 摘要类型
                - "brief": 简短摘要（1-2段）
                - "comprehensive": 全面摘要（包含所有关键条款）
                - "key_points": 关键点列表
                
        Returns:
            摘要文本
        """
        if not self.documents:
            return "No documents loaded. Please load a contract first."
        
        # 获取要总结的文档
        """  if pdf_path and pdf_path in self.documents:
            docs_to_summarize = self.documents[pdf_path]
        else:
            docs_to_summarize = []
            for docs in self.documents.values():
                docs_to_summarize.extend(docs)
        """
        if pdf_path and pdf_path in self.documents:
            docs_to_summarize = self.documents[pdf_path]
        else:
        # 最近一份
            last_key = next(reversed(self.documents.keys()))
            docs_to_summarize = self.documents[last_key]

        # 根据类型选择提示模板
        if summary_type == "brief":
            prompt_template = """
            Provide a brief 1-2 paragraph summary of this rental contract.
            Focus on the most important terms: rent amount, duration, and key obligations.
            
            Contract content:
            {text}
            
            Brief Summary:
            """
        elif summary_type == "key_points":
            prompt_template = """
            Extract and list the key points from this rental contract.
            Format as a numbered list covering:
            1. Rental amount and payment terms
            2. Lease duration and dates
            3. Security deposit details
            4. Maintenance responsibilities
            5. Termination conditions
            6. Important restrictions or rules
            7. Any special clauses
            
            Contract content:
            {text}
            
            Key Points:
            """
        else:  # comprehensive
            prompt_template = """
            Provide a comprehensive summary of this rental contract.
            Include all important sections:
            - Parties and Property Details
            - Financial Terms (rent, deposits, fees)
            - Lease Period and Renewal
            - Responsibilities (tenant vs landlord)
            - Rules and Restrictions
            - Termination and Penalties
            - Special Conditions
            
            Contract content:
            {text}
            
            Comprehensive Summary:
            """
        
        # 创建总结链
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        with get_openai_callback() as cb:
            if len(docs_to_summarize) > 10:
                # 长文档使用map_reduce策略
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="map_reduce",
                    map_prompt=prompt,
                    combine_prompt=prompt
                )
            else:
                # 短文档使用stuff策略
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="stuff",
                    prompt=prompt
                )
            
            summary = chain.run(docs_to_summarize)
            
            print(f"📊 Summary generated - Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
        
        return summary
    
    def ask_question(self, question: str, use_compression: bool = True) -> Dict:

        if not self.vectorstore:
            return {
                "answer": "No contract loaded. Please upload a PDF contract first.",
                "sources": []
            }

        # 选择检索器（是否开启压缩）
        if use_compression:
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.retriever
            )
        else:
            retriever = self.retriever

        # ⭐ 不把 memory 交给链；改为手动传 chat_history
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
            output_key="answer"  # 主输出为 answer
        )

        # 从本地 memory 取历史，传给链（list[BaseMessage] / list[str] 均可）
        try:
            history_vars = self.memory.load_memory_variables({})
            chat_history = history_vars.get("chat_history", [])
        except Exception:
            chat_history = []

        # 执行
        with get_openai_callback() as cb:
            result = qa_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })

        # 手动把本轮问答写回 memory（只存 question/answer）
        try:
            self.memory.save_context({"question": question}, {"answer": result.get("answer", "")})
        except Exception:
            pass

        # 整理来源
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": (doc.page_content[:200] + "...") if doc.page_content else "",
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            })

        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "tokens_used": cb.total_tokens if "cb" in locals() else 0
        }

    
    
    def compare_contracts(self, pdf_path1: str, pdf_path2: str) -> str:
        """
        比较两份合同的差异
        
        Args:
            pdf_path1: 第一份合同路径
            pdf_path2: 第二份合同路径
            
        Returns:
            比较结果
        """
        if pdf_path1 not in self.documents or pdf_path2 not in self.documents:
            return "Both contracts must be loaded first."
        
        prompt = PromptTemplate(
            template="""
            Compare these two rental contracts and highlight the key differences:
            
            Contract 1:
            {contract1}
            
            Contract 2:
            {contract2}
            
            Provide a detailed comparison covering:
            1. Rent amount differences
            2. Lease term differences
            3. Deposit variations
            4. Different rules or restrictions
            5. Maintenance responsibility changes
            6. Any other significant differences
            
            Comparison:
            """,
            input_variables=["contract1", "contract2"]
        )
        
        # 获取两份合同的摘要
        summary1 = self.summarize_contract(pdf_path1, "comprehensive")
        summary2 = self.summarize_contract(pdf_path2, "comprehensive")
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        comparison = chain.run(contract1=summary1, contract2=summary2)
        
        return comparison
    
    def extract_key_information(self) -> Dict:
        """
        提取合同关键信息到结构化格式
        
        Returns:
            包含关键信息的字典
        """
        if not self.vectorstore:
            return {"error": "No contract loaded"}
        
        # 定义要提取的关键信息
        extraction_queries = {
            "rent_amount": "What is the monthly rent amount?",
            "lease_duration": "What is the lease duration or term?",
            "security_deposit": "What is the security deposit amount?",
            "payment_due_date": "When is rent due each month?",
            "late_fee": "What is the late payment fee or penalty?",
            "pet_policy": "What is the pet policy?",
            "maintenance": "What are the maintenance responsibilities?",
            "termination": "What are the early termination conditions?",
            "utilities": "Who is responsible for utilities?",
            "parking": "What are the parking arrangements?"
        }
        
        extracted_info = {}
        
        for key, query in extraction_queries.items():
            result = self.ask_question(query, use_compression=True)
            extracted_info[key] = result["answer"]
        
        return extracted_info
    
    def clear_memory(self):
        """清除对话历史"""
        self.memory.clear()
        print("🧹 Conversation memory cleared")
    
    def save_vectorstore(self, path: str = "vectorstore"):
        """保存向量存储到磁盘"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"💾 Vector store saved to {path}")
    
    
    
    # 在 langchain_rag_system.py 中修改 load_vectorstore 方法

    def load_vectorstore(self, path: str = "vectorstore", allow_dangerous_deserialization: bool = False):
        """从磁盘加载向量存储
        
        Args:
            path: 向量存储路径
            allow_dangerous_deserialization: 是否允许加载pickle文件（仅在确信文件安全时使用）
        """
        if os.path.exists(path):
            # 新版本LangChain需要显式允许反序列化
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=allow_dangerous_deserialization
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10
                }
            )
            print(f"📂 Vector store loaded from {path}")
        else:
            print(f"⚠️ Vector store path not found: {path}")

    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        total_chunks = sum(len(docs) for docs in self.documents.values())
        
        return {
            "loaded_contracts": len(self.documents),
            "total_chunks": total_chunks,
            "vector_store_size": self.vectorstore.index.ntotal if self.vectorstore else 0,
            "memory_size": len(self.memory.buffer) if hasattr(self.memory, 'buffer') else 0,
            "contracts": list(self.contract_metadata.values())
        }
    
    # 在 langchain_rag_system.py 的 AdvancedContractRAG 类中添加以下方法

    def clear_all_documents(self):
        """清空所有已加载的文档和向量存储
        在加载新文件前调用，确保不会混合不同的合同
        """
        # 清空文档
        self.documents.clear()
        self.contract_metadata.clear()
        
        # 清空向量存储
        self.vectorstore = None
        self.retriever = None
        
        # 清空对话记忆
        if hasattr(self, 'memory') and self.memory:
            self.memory.clear()
        
        print("🧹 Cleared all documents and vector stores")

    def get_current_documents_info(self):
        """获取当前加载的文档信息"""
        if not self.documents:
            return "No documents loaded"
        
        info = []
        for doc_path, chunks in self.documents.items():
            info.append(f"📄 {Path(doc_path).name}: {len(chunks)} chunks")
        
        return "\n".join(info)

    def ensure_single_document(self, file_path: str):
        """确保只有一个文档被加载
        
        Args:
            file_path: 要加载的文件路径
        """
        # 检查是否是同一个文件
        if len(self.documents) == 1 and str(file_path) in self.documents:
            print(f"✅ Same document already loaded: {Path(file_path).name}")
            return True
        
        # 如果是不同文件，清空之前的
        if self.documents and str(file_path) not in self.documents:
            print(f"🔄 Different document detected, clearing previous data...")
            self.clear_all_documents()
        
        return False
        
        


# 使用示例
if __name__ == "__main__":
    #from config import OPENAI_API_KEY, OPENAI_MODEL
    
    api_key =os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # 初始化系统
    rag = AdvancedContractRAG(api_key, model)
    

    # 加载PDF
    result = rag.load_pdf("documents/contract.pdf")
    print(result)
    
    # 生成摘要
    summary = rag.summarize_contract(summary_type="key_points")
    print("\n📝 Contract Summary:")
    print(summary)
    
    # 问答
    question = "What is the monthly rent and when is it due?"
    answer = rag.ask_question(question)
    print(f"\n❓ Q: {question}")
    print(f"💡 A: {answer['answer']}")
    
    # 提取关键信息
    key_info = rag.extract_key_information()
    print("\n📊 Key Information:")
    for key, value in key_info.items():
        print(f"  {key}: {value}")


