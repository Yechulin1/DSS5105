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
import pandas as pd
# LangChain核心组件
from langchain_community.document_loaders.pdf import PyMuPDFLoader, PDFPlumberLoader
# 如只用其一，也可只留一个

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import (
    RetrievalQA, 
    ConversationalRetrievalChain,
    LLMChain
)
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.callbacks.manager import get_openai_callback

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
        
        
        # 设置代理（如果需要）
        proxies = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        if proxies:
            # 对于需要代理的情况，可以设置环境变量
            os.environ["OPENAI_PROXY"] = proxies

        # 初始化OpenAI组件 - 使用兼容的参数
        self.llm = ChatOpenAI(
            temperature=0,  # 降低到0提高速度
            model_name=model,  # 使用 model_name 而不是 model
            openai_api_key=api_key,
            max_tokens=400,  # 减少token数量以加快响应
            request_timeout=30,  # 减少超时时间
            streaming=False  # 禁用流式传输以获得完整响应
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key
        )
        
        # 文本分割器 - 智能分块（优化：减小块大小提高检索速度）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # 减小块大小以加快检索
            chunk_overlap=100,     # 减少重叠以提高速度
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]  # 支持中英文
        )
        
        # 向量存储
        self.vectorstore = None
        self.retriever = None
        
        # 对话记忆（优化：减少记忆轮数以加快处理）
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",   # ✅ 告诉 memory：输入字段叫 question
            output_key="answer",
            return_messages=True,
            k=3  # 减少到3轮对话以提高速度
        )
        
        # 存储已加载的文档
        self.documents = {}
        self.contract_metadata = {}
        
        # 缓存目录
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _normalize_text(self, text: str) -> str:
        """
        标准化文本中的Unicode字符
        将数学斜体等特殊字符转换为普通ASCII
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        import unicodedata
        
        # 数学斜体字符映射 (U+1D400-U+1D7FF)
        math_italic_lowercase = {
            '𝑎': 'a', '𝑏': 'b', '𝑐': 'c', '𝑑': 'd', '𝑒': 'e', '𝑓': 'f',
            '𝑔': 'g', '𝘩': 'h', '𝑖': 'i', '𝑗': 'j', '𝑘': 'k', '𝑙': 'l',
            '𝑚': 'm', '𝑛': 'n', '𝑜': 'o', '𝑝': 'p', '𝑞': 'q', '𝑟': 'r',
            '𝑠': 's', '𝑡': 't', '𝑢': 'u', '𝑣': 'v', '𝑤': 'w', '𝑥': 'x',
            '𝑦': 'y', '𝑧': 'z',
        }
        
        math_italic_uppercase = {
            '𝐴': 'A', '𝐵': 'B', '𝐶': 'C', '𝐷': 'D', '𝐸': 'E', '𝐹': 'F',
            '𝐺': 'G', '𝐻': 'H', '𝐼': 'I', '𝐽': 'J', '𝐾': 'K', '𝐿': 'L',
            '𝑀': 'M', '𝑁': 'N', '𝑂': 'O', '𝑃': 'P', '𝑄': 'Q', '𝑅': 'R',
            '𝑆': 'S', '𝑇': 'T', '𝑈': 'U', '𝑉': 'V', '𝑊': 'W', '𝑋': 'X',
            '𝑌': 'Y', '𝑍': 'Z',
        }
        
        # 其他特殊字符
        special_chars = {
            'ℎ': 'h',   # PLANCK CONSTANT
            '℘': 'P',   # SCRIPT CAPITAL P
            'ℓ': 'l',   # SCRIPT SMALL L
            'ℯ': 'e',   # SCRIPT SMALL E
            'ℊ': 'g',   # SCRIPT SMALL G
            'ℴ': 'o',   # SCRIPT SMALL O
        }
        
        # 合并所有映射
        char_map = {**math_italic_lowercase, **math_italic_uppercase, **special_chars}
        
        # 逐字符替换
        result = []
        for char in text:
            result.append(char_map.get(char, char))
        
        text = ''.join(result)

        text = text.replace('$', 'S$')  # 或者直接删除这一行


        # Unicode标准化（NFKD：兼容分解）
        text = unicodedata.normalize('NFKD', text)
        
        # 可选：移除不可见控制字符
        text = ''.join(c for c in text if c.isprintable() or c.isspace())
        
        return text
    
    def _normalize_documents(self, documents):
        """
        标准化文档列表中的所有文本
        
        Args:
            documents: LangChain Document对象列表
            
        Returns:
            标准化后的Document对象列表
        """
        from langchain.schema import Document
        
        normalized_docs = []
        for doc in documents:
            normalized_text = self._normalize_text(doc.page_content)
            normalized_doc = Document(
                page_content=normalized_text,
                metadata=doc.metadata
            )
            normalized_docs.append(normalized_doc)
        
        return normalized_docs
       
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
        

        documents = self._normalize_documents(documents)
        
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
            
            # 创建增强检索器（优化：减少检索数量以加快速度）
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # 使用相似度搜索，速度更快
                search_kwargs={
                    "k": 8,  # 返回5个最相关的块
                    #"fetch_k": 10  # 先获取10个候选
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
                - "key points": 关键点列表
                
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
        
        docs_to_summarize = self._normalize_documents(docs_to_summarize)
        
        # 优化：限制文档块数量以提高速度（所有类型统一）
        if len(docs_to_summarize) > 10:
            docs_to_summarize = docs_to_summarize[:10]
            print(f"📄 Optimized: Using first 10 chunks for faster processing")
        
        # 根据类型选择提示模板
        if summary_type == "brief":
            prompt_template = """
            Provide a brief but informative summary of this rental contract in 2-3 short paragraphs.
            
            Paragraph 1: Property details, parties involved, and lease duration
            Paragraph 2: Financial terms - rent amount, payment schedule, security deposit, and any fees
            Paragraph 3: Key responsibilities and important rules/restrictions
            
            Use specific numbers and dates from the contract.
            
            Contract content:
            {text}
            
            Brief Summary:
            """
        elif summary_type == "key points":
            prompt_template = """
            Extract key points from this rental contract in a numbered list.
            Be concise - one line per point:
            1. Rent amount and due date
            2. Lease start and end dates
            3. Security deposit amount
            4. Tenant maintenance duties
            5. Landlord maintenance duties
            6. Termination notice period
            7. Key restrictions (pets, smoking, etc.)
            
            Contract content:
            {text}
            
            Key Points:
            """
        else:  # comprehensive
            prompt_template = """
            Provide a concise comprehensive summary of this rental contract in 300 words or less.
            Use structured format with key details only.
            
            Format:
            **PARTIES & PROPERTY**
            [Names, address in 1 line]
            
            **FINANCIAL TERMS**
            • Rent: [amount/month]
            • Deposit: [amount]
            • Payment: [due date]
            • Late fee: [amount/terms if any]
            • Fees: [if any]
            
            **LEASE PERIOD**
            [Start] to [End] | Renewal: [terms]
            
            **RESPONSIBILITIES**
            Landlord: [key duties]
            Tenant: [key duties]
            
            **RULES & RESTRICTIONS**
            [Bullet list of key rules]
            • Pets: [policy]
            
            **TERMINATION**
            Notice: [period] | Conditions: [brief]
            
            **SPECIAL TERMS**
            [Any unique clauses, if none write "None"]
            
            Be brief and use exact numbers/dates from contract.
            
            Contract content:
            {text}
            
            Comprehensive Summary:
            """
        
        # 创建总结链 - 统一使用stuff策略（最快）
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        with get_openai_callback() as cb:
            # 统一使用stuff策略，最快速
            chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            summary = chain.run(docs_to_summarize)
            
            print(f"📊 Summary generated - Tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
        
        return summary
    
    def ask_question(self, question: str, use_compression: bool = False) -> Dict:
        """
        优化版问答：默认关闭压缩以提高速度
        """
        if not self.vectorstore:
            return {
                "answer": "No contract loaded. Please upload a PDF contract first.",
                "sources": []
            }

        # 选择检索器（压缩会显著降低速度，默认关闭）
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

        # ⭐ 改进的来源匹配逻辑：根据答案内容筛选最相关的来源
        answer_text = result.get("answer", "")
        source_documents = result.get("source_documents", [])
        
        # 如果没有明确答案或来源，返回空
        if not answer_text or not source_documents:
            return {
                "answer": answer_text,
                "sources": [],
                "tokens_used": cb.total_tokens if "cb" in locals() else 0
            }
        
        # 提取答案中的关键信息（数字、金额、日期等）
        import re
        answer_keywords = set()
        
        # 提取数字（包括金额）
        numbers = re.findall(r'\$?\d+[,\d]*\.?\d*', answer_text)
        answer_keywords.update(numbers)
        
        # 提取日期
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', answer_text, re.IGNORECASE)
        answer_keywords.update(dates)
        
        # 提取答案中的重要词汇（长度>3的单词，排除常见词）
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'will', 'with'}
        words = re.findall(r'\b[A-Za-z]{4,}\b', answer_text.lower())
        answer_keywords.update([w for w in words if w not in stopwords])
        
        # 对每个来源文档计算相关性分数
        scored_sources = []
        for doc in source_documents:
            content = doc.page_content if doc.page_content else ""
            content_lower = content.lower()
            
            # 计算匹配分数
            score = 0
            matched_keywords = []
            
            for keyword in answer_keywords:
                keyword_lower = keyword.lower()
                # 精确匹配得高分
                if keyword_lower in content_lower:
                    # 数字和金额匹配得更高分
                    if re.match(r'\$?\d+', keyword):
                        score += 10
                    # 日期匹配得高分
                    elif re.search(r'\d{1,2}[/-]\d{1,2}', keyword):
                        score += 8
                    else:
                        score += 2
                    matched_keywords.append(keyword)
            
            # 只保留有匹配的文档
            if score > 0:
                scored_sources.append({
                    "score": score,
                    "content": content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "matched_keywords": matched_keywords
                })
        
        # 如果没有匹配的来源，返回分数最高的前3个原始来源
        if not scored_sources:
            sources = []
            for doc in source_documents[:3]:  # 最多3个
                sources.append({
                    "content": doc.page_content if doc.page_content else "",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "similarity_score": 0  # 没有匹配时分数为0
                })
        else:
            # 按分数排序
            scored_sources.sort(key=lambda x: x["score"], reverse=True)
            
            # 只保留相似度分数>=20的来源
            filtered_sources = [src for src in scored_sources if src["score"] >= 20]
            
            sources = []
            
            # 如果过滤后没有符合条件的来源，返回分数最高的前3个原始来源
            if not filtered_sources:
                for doc in source_documents[:3]:  # 最多3个
                    sources.append({
                        "content": doc.page_content if doc.page_content else "",
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "similarity_score": 0  # 没有匹配时分数为0
                    })
            else:
                # 如果过滤后的来源少于3个，只返回最相关的一个
                # 如果有3个或更多，则返回前3个
                num_sources = 1 if len(filtered_sources) < 3 else 3
                
                for src in filtered_sources[:num_sources]:
                    sources.append({
                        "content": src["content"],
                        "source": src["source"],
                        "page": src["page"],
                        "similarity_score": src["score"]
                    })
        
        # 如果只有一个明确信息，只返回1个来源
        # 检测答案是否只包含单一信息（例如只有一个数字或日期）
        if len(numbers) == 1 and len(dates) == 0 and len(sources) > 1:
            # 只保留分数最高的那个
            sources = sources[:1]
        
        return {
            "answer": answer_text,
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
        提取合同关键信息到结构化格式（优先从摘要提取）
        
        Returns:
            包含关键信息的字典
        """
        # 若未加载向量库，但已有文档，也可直接生成摘要
        if not self.documents:
            return {"error": "No contract loaded"}

        # 先生成综合摘要
        summary_text = self.summarize_contract(summary_type="comprehensive")

        # 基于摘要的结构化提取（JSON输出）
        template = """
        You are extracting key information from a rental contract summary.
        Only use the provided Summary content. Do not assume missing details.
        Use specific numbers and dates from the Summary.
        If a field is not present, return exactly "Not mentioned".

        Summary:
        {summary}

        Extract and return a compact JSON object with these keys:
        - rent_amount: string
        - lease_duration: string
        - security_deposit: string
        - payment_due_date: string
        - late_fee: string
        - pet_policy: string
        - maintenance: string
        - termination: string
        - utilities: string
        - parking: string
        """

        prompt = PromptTemplate(template=template, input_variables=["summary"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        raw = chain.run(summary=summary_text)

        # 尝试解析JSON；失败则回退为全部字段"Not mentioned"
        import json, re
        try:
            # 可能模型返回包含代码块，先抽取JSON片段
            match = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(match.group(0) if match else raw)
        except Exception:
            data = {}

        # 统一字段与回退值
        keys = [
            "rent_amount","lease_duration","security_deposit","payment_due_date",
            "late_fee","pet_policy","maintenance","termination","utilities","parking"
        ]
        extracted_info = {k: (str(data.get(k, "")).strip() or "Not mentioned") for k in keys}

        # 对缺失项进行检索式回填（若向量库可用）
        if self.vectorstore:
            fallback_queries = {
                "rent_amount": "What is the monthly rent amount? Use exact amount.",
                "lease_duration": "What is the lease duration? Use exact months/years.",
                "security_deposit": "What is the security deposit amount? Use exact amount.",
                "payment_due_date": "On what date each month is rent due? Use exact day/date.",
                "late_fee": "What is the late payment fee or penalty? Use exact amount/terms.",
                "pet_policy": "What is the pet policy? Are pets allowed? State policy briefly.",
                "maintenance": "What are landlord and tenant maintenance responsibilities? Summarize briefly.",
                "termination": "What are the lease termination or early termination conditions?",
                "utilities": "Who pays utilities (water, electricity, gas, etc.)?",
                "parking": "What parking arrangements or spaces are provided?"
            }

            for k, v in extracted_info.items():
                if v == "Not mentioned":
                    qa = self.ask_question(fallback_queries[k], use_compression=False)
                    ans = qa.get("answer", "").strip()
                    if ans and ans.lower() not in {"not mentioned", "unknown", "not specified"}:
                        extracted_info[k] = self._simplify_answer(ans, k)

        return extracted_info
    

    def extract_key_information_parallel(self) -> Dict:
        """
        基于摘要的单次结构化提取（更快更稳定）
        
        Returns:
            包含关键信息的字典
        """
        if not self.documents:
            return {"error": "No contract loaded"}

        # 复用综合摘要 + JSON提取（与非并行版本一致）
        summary_text = self.summarize_contract(summary_type="comprehensive")
        template = """
        From the Summary below, extract key rental contract information.
        Use exact numbers/dates only from the Summary. If missing, return "Not mentioned".

        Summary:
        {summary}

        Return JSON with keys:
        rent_amount, lease_duration, security_deposit, payment_due_date,
        late_fee, pet_policy, maintenance, termination, utilities, parking
        """
        prompt = PromptTemplate(template=template, input_variables=["summary"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        raw = chain.run(summary=summary_text)

        import json, re
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(match.group(0) if match else raw)
        except Exception:
            data = {}

        keys = [
            "rent_amount","lease_duration","security_deposit","payment_due_date",
            "late_fee","pet_policy","maintenance","termination","utilities","parking"
        ]
        info = {k: (str(data.get(k, "")).strip() or "Not mentioned") for k in keys}

        # 并行版本也进行缺失项回填（若向量库可用）
        if self.vectorstore:
            fallback_queries = {
                "rent_amount": "What is the monthly rent amount? Use exact amount.",
                "lease_duration": "What is the lease duration? Use exact months/years.",
                "security_deposit": "What is the security deposit amount? Use exact amount.",
                "payment_due_date": "On what date each month is rent due? Use exact day/date.",
                "late_fee": "What is the late payment fee or penalty? Use exact amount/terms.",
                "pet_policy": "What is the pet policy? Are pets allowed? State policy briefly.",
                "maintenance": "What are landlord and tenant maintenance responsibilities? Summarize briefly.",
                "termination": "What are the lease termination or early termination conditions?",
                "utilities": "Who pays utilities (water, electricity, gas, etc.)?",
                "parking": "What parking arrangements or spaces are provided?"
            }

            # 顺序回填（避免API过多并发）
            for k, v in info.items():
                if v == "Not mentioned":
                    qa = self.ask_question(fallback_queries[k], use_compression=False)
                    ans = qa.get("answer", "").strip()
                    if ans and ans.lower() not in {"not mentioned", "unknown", "not specified"}:
                        info[k] = self._simplify_answer(ans, k)

        return info
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_key = {
                executor.submit(
                    self.ask_question, 
                    query, 
                    use_compression=False  # 关闭压缩，进一步提速
                ): key
                for key, query in extraction_queries.items()
            }
            
            # 收集结果（按完成顺序）
            completed = 0
            total = len(extraction_queries)
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    answer = result["answer"]
                    
                    # 后处理：将模糊或未知答案替换为 "Not mentioned"
                    answer_lower = answer.lower().strip()
                    uncertain_phrases = [
                        "i don't know",
                        "i do not know",
                        "not found",
                        "cannot find",
                        "unable to find",
                        "no information",
                        "not specified",
                        "not mentioned",
                        "not available",
                        "not provided",
                        "unclear",
                        "unknown"
                    ]
                    
                    # 检查是否是不确定的答案
                    if any(phrase in answer_lower for phrase in uncertain_phrases) or len(answer.strip()) < 3:
                        extracted_info[key] = "Not mentioned"
                    else:
                        # 简化答案，使其更简洁
                        simplified_answer = self._simplify_answer(answer, key)
                        extracted_info[key] = simplified_answer
                    
                    completed += 1
                    print(f"✅ [{completed}/{total}] Extracted: {key}")
                except Exception as e:
                    extracted_info[key] = "Not mentioned"
                    completed += 1
                    print(f"❌ [{completed}/{total}] Failed: {key} - {e}")
        
        elapsed = time.time() - start_time
        print(f"🎉 All extractions completed in {elapsed:.2f} seconds")
        
        return extracted_info

    def _simplify_answer(self, answer: str, key: str) -> str:
        """
        简化答案，使其更简洁，避免长句子，但保留关键细节
        
        Args:
            answer: 原始答案
            key: 字段键名
            
        Returns:
            简化的答案
        """
        import re
        
        # 如果答案已经是简短的，直接返回
        if len(answer.strip()) <= 60:
            return answer.strip()
        
        # 根据不同字段类型进行简化
        if key == "rent_amount":
            # 提取金额
            amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', answer)
            if amounts:
                return amounts[0]
            # 查找数字金额
            numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', answer)
            if numbers:
                return f"${numbers[0]}"
                
        elif key == "lease_duration":
            # 提取时间段
            durations = re.findall(r'\b\d+\s+(?:month|year|week|day)s?\b', answer, re.IGNORECASE)
            if durations:
                return durations[0]
            # 查找数字+时间单位
            time_patterns = re.findall(r'\b\d+\s*(?:month|year|week|day|yr|mo|wk|dy)s?\b', answer, re.IGNORECASE)
            if time_patterns:
                return time_patterns[0]
                
        elif key == "security_deposit":
            # 提取押金金额
            amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', answer)
            if amounts:
                return amounts[0]
            numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', answer)
            if numbers:
                return f"${numbers[0]}"
                
        elif key == "payment_due_date":
            # 提取日期
            dates = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\b', answer)
            if dates:
                return f"{dates[0]}th of each month"
            # 查找"first", "last"等
            day_words = re.findall(r'\b(?:first|last|1st|15th|30th|31st)\b', answer, re.IGNORECASE)
            if day_words:
                return f"{day_words[0].lower()} of month"
                
        elif key == "late_fee":
            # 提取罚款金额或百分比
            amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?%', answer)
            if amounts:
                return amounts[0]
            numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', answer)
            if numbers:
                return numbers[0] + ("%" if "%" in answer else "")
                
        elif key == "pet_policy":
            # 简化宠物政策，但保留关键细节
            if "not allowed" in answer.lower() or "no pets" in answer.lower():
                return "No pets allowed"
            elif "allowed" in answer.lower() or "permitted" in answer.lower():
                # 查找押金信息
                deposits = re.findall(r'\$[\d,]+(?:\.\d{2})?\s*(?:deposit|fee)', answer, re.IGNORECASE)
                if deposits:
                    return f"Pets allowed with {deposits[0]} deposit"
                else:
                    return "Pets allowed"
            elif "deposit" in answer.lower():
                deposits = re.findall(r'\$[\d,]+', answer)
                if deposits:
                    return f"Pet deposit: {deposits[0]}"
                    
        elif key == "utilities":
            # 保留 utilities 的具体细节
            utilities_mentioned = []
            
            # 查找常见的公用事业项目
            utility_types = ['water', 'electricity', 'gas', 'electric', 'power', 'heating', 'cooling', 'internet', 'cable', 'trash', 'sewage', 'garbage']
            
            for utility in utility_types:
                if utility in answer.lower():
                    utilities_mentioned.append(utility.title())
            
            if utilities_mentioned:
                # 确定谁负责
                if "tenant" in answer.lower() and "landlord" not in answer.lower():
                    return f"Tenant pays: {', '.join(utilities_mentioned)}"
                elif "landlord" in answer.lower() and "tenant" not in answer.lower():
                    return f"Landlord pays: {', '.join(utilities_mentioned)}"
                elif "shared" in answer.lower() or "split" in answer.lower():
                    return f"Shared: {', '.join(utilities_mentioned)}"
                elif "included" in answer.lower():
                    return f"Included in rent: {', '.join(utilities_mentioned)}"
                else:
                    return f"Utilities: {', '.join(utilities_mentioned)}"
            else:
                # 如果没找到具体项目，使用原有逻辑
                if "tenant" in answer.lower() and "landlord" not in answer.lower():
                    return "Tenant pays utilities"
                elif "landlord" in answer.lower() and "tenant" not in answer.lower():
                    return "Landlord pays utilities"
                elif "shared" in answer.lower() or "split" in answer.lower():
                    return "Utilities shared/split"
                elif "included" in answer.lower():
                    return "Utilities included in rent"
                
        elif key == "parking":
            # 保留停车的细节
            if "included" in answer.lower():
                return "Parking included"
            elif "available" in answer.lower():
                spaces = re.findall(r'\b\d+\s*(?:space|spot|car)s?\b', answer, re.IGNORECASE)
                if spaces:
                    return f"Parking available: {spaces[0]}"
                else:
                    return "Parking available"
            spaces = re.findall(r'\b\d+\s*(?:space|spot|car)s?\b', answer, re.IGNORECASE)
            if spaces:
                return spaces[0]
                
        elif key == "maintenance":
            # 保留维护责任的细节
            if "tenant" in answer.lower() and "landlord" not in answer.lower():
                return "Tenant responsible for maintenance"
            elif "landlord" in answer.lower() and "tenant" not in answer.lower():
                return "Landlord responsible for maintenance"
            elif "shared" in answer.lower():
                return "Maintenance responsibilities shared"
            # 尝试提取具体的维护项目
            maintenance_items = []
            maint_types = ['repairs', 'fixtures', 'appliances', 'plumbing', 'electrical', 'heating', 'cooling', 'painting']
            for item in maint_types:
                if item in answer.lower():
                    maintenance_items.append(item.title())
            if maintenance_items:
                return f"Maintenance: {', '.join(maintenance_items)}"
                
        elif key == "termination":
            # 保留终止条件的细节
            if "notice" in answer.lower():
                notices = re.findall(r'\b\d+\s*(?:day|week|month)s?\s*notice\b', answer, re.IGNORECASE)
                if notices:
                    return f"{notices[0]} notice required"
            # 查找提前终止条款
            early_terms = re.findall(r'(?:break|terminate|early).{0,50}(?:fee|penalty|charge)', answer, re.IGNORECASE)
            if early_terms:
                fees = re.findall(r'\$[\d,]+', early_terms[0])
                if fees:
                    return f"Early termination fee: {fees[0]}"
                    
        # 对于其他长答案，尝试更好地概括而不是简单截断
        simplified = answer.strip()
        if len(simplified) > 60:
            # 提取关键信息模式
            # 查找金额
            amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', simplified)
            # 查找百分比
            percentages = re.findall(r'\d+(?:\.\d+)?%', simplified)
            # 查找日期
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', simplified)
            # 查找时间段
            periods = re.findall(r'\b\d+\s+(?:month|year|week|day)s?\b', simplified, re.IGNORECASE)
            
            key_info = amounts + percentages + dates + periods
            
            if key_info:
                # 如果有关键信息，构建简洁摘要
                summary_parts = []
                if amounts:
                    summary_parts.append(f"Amount: {', '.join(amounts[:2])}")  # 最多显示2个金额
                if percentages:
                    summary_parts.append(f"Rate: {', '.join(percentages[:1])}")
                if dates:
                    summary_parts.append(f"Date: {dates[0]}")
                if periods:
                    summary_parts.append(f"Period: {periods[0]}")
                
                return "; ".join(summary_parts)
            else:
                # 如果没有关键信息，尝试提取前两个句子
                sentences = re.split(r'[.!?]+', simplified)
                meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10][:2]
                if meaningful_sentences:
                    return ". ".join(meaningful_sentences) + "."
                else:
                    # 最后手段：智能截断
                    return simplified[:55] + "..."
        
        return simplified

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
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    # 初始化系统
    rag = AdvancedContractRAG(api_key, model)
    

    # 加载PDF
    result = rag.load_pdf("documents/contract.pdf")
    print(result)
    
    # 生成摘要
    summary = rag.summarize_contract(summary_type="key points")
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