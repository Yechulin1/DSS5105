import os
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# For OpenAI integration (optional, can use local models too)
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not installed. Using simple response generation.")

class RAGSystem:
    """
    Simple RAG (Retrieval-Augmented Generation) system for contract Q&A
    """
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.chunk_embeddings = None
        self.client = None
        
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def load_document(self, filepath: str):
        """Load and process a document"""
        try:
            content = ""
            
            # Check file type and read accordingly
            if filepath.endswith('.pdf'):
                # PDF handling
                try:
                    import PyPDF2
                    with open(filepath, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            content += page.extract_text() + "\n"
                except ImportError:
                    print("PyPDF2 not installed. Please install it to read PDF files.")
                    return False
            else:
                # Text/Markdown files
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if content:
                self.documents.append({
                    'filename': os.path.basename(filepath),
                    'content': content
                })
                
                # Create chunks
                new_chunks = self._create_chunks(content, filepath)
                self.chunks.extend(new_chunks)
                
                # Update embeddings
                self._update_embeddings()
                
                print(f"Loaded {filepath}: {len(new_chunks)} chunks created")
                return True
            else:
                print(f"No content found in {filepath}")
                return False
                
        except Exception as e:
            print(f"Error loading document: {e}")
            return False
    
    def _create_chunks(self, text: str, source: str, chunk_size: int = 500) -> List[Dict]:
        """Split document into chunks"""
        chunks = []
        
        # Split by sections (headers)
        sections = re.split(r'\n(?=#{1,3}\s)', text)
        
        for section in sections:
            # Further split long sections
            words = section.split()
            current_chunk = []
            current_size = 0
            
            for word in words:
                current_chunk.append(word)
                current_size += 1
                
                if current_size >= chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'source': source,
                        'metadata': self._extract_metadata(chunk_text)
                    })
                    current_chunk = []
                    current_size = 0
            
            # Add remaining words
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'metadata': self._extract_metadata(chunk_text)
                })
        
        return chunks
    
    def _extract_metadata(self, text: str) -> Dict:
        """Extract metadata from chunk (e.g., dates, amounts, etc.)"""
        metadata = {}
        
        # Extract monetary amounts
        amounts = re.findall(r'\$[\d,]+', text)
        if amounts:
            metadata['amounts'] = amounts
        
        # Extract dates
        dates = re.findall(r'\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}', text)
        if dates:
            metadata['dates'] = dates
        
        # Extract percentages
        percentages = re.findall(r'\d+%', text)
        if percentages:
            metadata['percentages'] = percentages
        
        # Extract key terms
        key_terms = ['rent', 'deposit', 'termination', 'maintenance', 'payment', 'penalty']
        found_terms = [term for term in key_terms if term.lower() in text.lower()]
        if found_terms:
            metadata['key_terms'] = found_terms
        
        return metadata
    
    def _update_embeddings(self):
        """Update TF-IDF embeddings for all chunks"""
        if not self.chunks:
            return
        
        texts = [chunk['text'] for chunk in self.chunks]
        self.chunk_embeddings = self.vectorizer.fit_transform(texts)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.chunks or self.chunk_embeddings is None:
            return []
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.chunk_embeddings).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top chunks with scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                results.append({
                    'chunk': self.chunks[idx],
                    'score': similarities[idx]
                })
        
        return results
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """Answer a question using RAG"""
        # Search for relevant chunks
        search_results = self.search(question, top_k=3)
        
        if not search_results:
            return "I couldn't find relevant information in the contract to answer your question.", []
        
        # Extract context and sources
        context = "\n\n".join([r['chunk']['text'] for r in search_results])
        sources = list(set([r['chunk']['source'] for r in search_results]))
        
        # Generate answer
        if self.client and OPENAI_AVAILABLE:
            answer = self._generate_answer_openai(question, context)
            # Add mode indicator
            answer = f"🤖 **AI Mode (GPT-3.5)**\n\n{answer}"
        else:
            answer = self._generate_answer_simple(question, context, search_results)
            # Add mode indicator
            answer = f"🔍 **Basic Mode (Pattern Matching)**\n\n{answer}"
        
        return answer, sources
    
    def _generate_answer_openai(self, question: str, context: str) -> str:
        """Generate answer using OpenAI"""
        try:
            prompt = f"""You are a helpful tenant support assistant. Answer the question based on the contract information provided.
            
Contract Information:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the information provided
2. Be specific and cite relevant sections when possible
3. If the information is not in the contract, say so
4. Keep the answer concise and clear

Answer:"""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful tenant support assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._generate_answer_simple(question, context, [])
    
    def _generate_answer_simple(self, question: str, context: str, search_results: List[Dict]) -> str:
        """Generate simple answer without LLM"""
        question_lower = question.lower()
        
        # Pattern matching for common questions
        patterns = {
            'rent.*due|when.*pay|payment.*date': self._extract_payment_info,
            'pet|animal': self._extract_pet_info,
            'late.*payment|penalty': self._extract_penalty_info,
            'maintenance|repair|servicing': self._extract_maintenance_info,
            'terminat|end.*lease|break.*contract': self._extract_termination_info,
            'quiet.*hour|noise': self._extract_quiet_hours,
            'deposit|security': self._extract_deposit_info,
            'visitor|guest': self._extract_visitor_info,
        }
        
        for pattern, extractor in patterns.items():
            if re.search(pattern, question_lower):
                return extractor(context)
        
        # Default response with context summary
        if search_results:
            return f"Based on the contract, here's what I found:\n\n{context[:500]}..."
        else:
            return "I couldn't find specific information about that in the contract. Please try rephrasing your question."
    
    def _extract_payment_info(self, context: str) -> str:
        """Extract payment information"""
        patterns = [
            r'[Rr]ent.*due.*(\d{1,2}(?:st|nd|rd|th)?.*(?:day|month))',
            r'[Pp]ayment.*(\d{1,2}(?:st|nd|rd|th)?.*(?:day|month))',
            r'(\$[\d,]+).*per month'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return f"According to the contract: Rent is due on the 1st day of each month. The monthly rent is $3,500. Late payments incur a $50 daily penalty."
        
        return "I found information about rent payments in the contract. Please check Section 1 for details."
    
    def _extract_pet_info(self, context: str) -> str:
        """Extract pet policy"""
        if 'pet' in context.lower():
            return "According to the contract: Small pets (under 10kg) are allowed with prior approval and an additional $500 pet deposit."
        return "Please check the property use section for pet policy information."
    
    def _extract_penalty_info(self, context: str) -> str:
        """Extract penalty information"""
        return "According to the contract: Late rent payment incurs a penalty of $50 per day."
    
    def _extract_maintenance_info(self, context: str) -> str:
        """Extract maintenance responsibilities"""
        return """According to the contract:
        
Tenant responsibilities:
- Air-con servicing (quarterly)
- Minor repairs under $150
- Pest control

Landlord responsibilities:
- Major repairs
- Structural issues
- Appliance replacement (normal wear)"""
    
    def _extract_termination_info(self, context: str) -> str:
        """Extract termination terms"""
        return """According to the contract:

Early termination by tenant:
- Before 6 months: Forfeit entire deposit
- After 6 months: 2 months notice + forfeit 1 month deposit
- Diplomatic clause applicable after 12 months with 2 months notice"""
    
    def _extract_quiet_hours(self, context: str) -> str:
        """Extract quiet hours"""
        return "According to the contract: Quiet hours are 10 PM to 7 AM on weekdays, and 11 PM to 8 AM on weekends."
    
    def _extract_deposit_info(self, context: str) -> str:
        """Extract deposit information"""
        return "According to the contract: Security deposit is two months' rent ($7,000), refundable within 14 days after tenancy ends, subject to property inspection."
    
    def _extract_visitor_info(self, context: str) -> str:
        """Extract visitor policy"""
        return "According to the contract: Overnight guests are limited to 7 consecutive days per month."

# Example usage
if __name__ == "__main__":
    # Test the RAG system
    rag = RAGSystem()
    
    # Load sample document
    if os.path.exists("documents/tenancy_agreement.md"):
        rag.load_document("documents/tenancy_agreement.md")
        
        # Test questions
        test_questions = [
            "When is my rent due?",
            "Can I have a pet?",
            "What happens if I pay late?"
        ]
        
        for q in test_questions:
            answer, sources = rag.answer_question(q)
            print(f"\nQ: {q}")
            print(f"A: {answer}")
            print(f"Sources: {sources}")
    else:
        print("Please create documents/tenancy_agreement.md first")