"""
RentalPeace 
evaluation module
Author: cipher-h
Date: 2025-11-13
Version: 4.0 (Latest - With Quick Questions)
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ROUGE Score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️ ROUGE Score not available. Install with: pip install rouge-score")

# RAG System
from langchain_rag_system import AdvancedContractRAG


class RentalPeaceEvaluator:
    """
    RentalPeace 

    Evaluation Dimensions：
    1. Retrieval Quality
    2. Answer Quality
    3. Quick Question Buttons Quality
    4. Source Citation Quality
    5. Summary Quality
    6. Extraction Accuracy
    7. System Efficiency
    """
    
    def __init__(self, rag_system: AdvancedContractRAG, evaluation_mode: str = "fast"):
        """
        Initialize the evaluator
        
        Args:
            rag_system: AdvancedContractRAG 
            evaluation_mode: ("fast", "accurate")
        """
        self.rag = rag_system
        self.evaluation_mode = evaluation_mode

        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )

        # Initialize evaluation results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_mode": evaluation_mode,
            "evaluator": "cipher-h",
            "retrieval_quality": {},
            "answer_quality": {},
            "quick_questions_quality": {},  
            "source_citation": {},
            "summary_quality": {},
            "extraction_accuracy": {},
            "efficiency": {},
        }
    
    # ========================================
    # 1. Retrieval Quality Evaluation
    # ========================================
    
    def evaluate_retrieval_quality(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate retrieval quality
        
        Args:
            test_cases: Retrieval test cases
                Format: [
                    {
                        "question": "What is the monthly rent?",
                        "expected_keywords": ["rent", "monthly", "7500"],
                        "expected_page": 1,
                        "difficulty": "easy"
                    }
                ]
        
        Returns:
            Retrieval quality metrics dictionary
        """
        print("\n" + "="*60)
        print("🎯 1. RETRIEVAL QUALITY EVALUATION")
        print("="*60)
        
        if not test_cases:
            print("⚠️ No test cases provided")
            return {}
        
        mrr_scores = []
        recall_at_k_scores = {1: [], 3: [], 5: []}
        keyword_coverage_scores = []
        
        for i, test in enumerate(test_cases, 1):
            question = test["question"]
            expected_keywords = test.get("expected_keywords", [])
            expected_page = test.get("expected_page")
            
            print(f"\n📝 Test {i}/{len(test_cases)}: {question[:60]}...")
            
            try:

                docs = self.rag.retriever.get_relevant_documents(question)

                # Calculate MRR (Mean Reciprocal Rank)
                reciprocal_rank = 0
                for rank, doc in enumerate(docs, 1):
                    doc_page = doc.metadata.get("page", -1)
                    if expected_page and doc_page == expected_page:
                        reciprocal_rank = 1.0 / rank
                        print(f"   ✅ Found at rank {rank} (MRR: {reciprocal_rank:.3f})")
                        break
                
                if reciprocal_rank == 0 and expected_page:
                    print(f"   ❌ Expected page {expected_page} not found")
                
                mrr_scores.append(reciprocal_rank)

                # Calculate Recall@K
                for k in [1, 3, 5]:
                    top_k_docs = docs[:k]
                    top_k_pages = [d.metadata.get("page") for d in top_k_docs]
                    
                    if expected_page:
                        recall = 1 if expected_page in top_k_pages else 0
                        recall_at_k_scores[k].append(recall)
                
                # Calculate Keyword Coverage
                if expected_keywords and docs:
                    top_doc_content = docs[0].page_content.lower()
                    keywords_found = sum(1 for kw in expected_keywords 
                                       if kw.lower() in top_doc_content)
                    coverage = keywords_found / len(expected_keywords)
                    keyword_coverage_scores.append(coverage)
                    print(f"   📌 Keywords: {keywords_found}/{len(expected_keywords)} ({coverage:.1%})")
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
                mrr_scores.append(0)
                for k in [1, 3, 5]:
                    recall_at_k_scores[k].append(0)
        
        # Summarize results
        results = {
            "mrr": np.mean(mrr_scores) if mrr_scores else 0,
            "recall@1": np.mean(recall_at_k_scores[1]) if recall_at_k_scores[1] else 0,
            "recall@3": np.mean(recall_at_k_scores[3]) if recall_at_k_scores[3] else 0,
            "recall@5": np.mean(recall_at_k_scores[5]) if recall_at_k_scores[5] else 0,
            "keyword_coverage": np.mean(keyword_coverage_scores) if keyword_coverage_scores else 0,
            "total_tests": len(test_cases)
        }
        
        print("\n" + "-"*60)
        print("📊 RETRIEVAL SUMMARY:")
        print(f"   MRR: {results['mrr']:.3f}")
        print(f"   Recall@1: {results['recall@1']:.1%}")
        print(f"   Recall@3: {results['recall@3']:.1%}")
        print(f"   Recall@5: {results['recall@5']:.1%}")
        print(f"   Keyword Coverage: {results['keyword_coverage']:.1%}")
        print("-"*60)
        
        self.results["retrieval_quality"] = results
        return results
    
    # ========================================
    # 2. Answer Quality Evaluation
    # ========================================
    
    def evaluate_answer_quality(self, qa_pairs: List[Dict]) -> Dict:
        """
        Evaluate answer quality
        
        Args:
            qa_pairs: Question-answer pairs
                Format: [
                    {
                        "question": "What is the monthly rent?",
                        "reference_answer": "The monthly rent is S$7,500.",
                        "category": "financial",
                        "difficulty": "easy"
                    }
                ]
        
        Returns:
            Answer quality metrics dictionary
        """
        print("\n" + "="*60)
        print(f"💬 2. ANSWER QUALITY EVALUATION (Mode: {self.evaluation_mode.upper()})")
        print("="*60)
        
        if not qa_pairs:
            print("⚠️ No QA pairs provided")
            return {}
        
        predictions = []
        references = []
        faithfulness_scores = []
        answer_lengths = []
        response_times = []
        
        for i, pair in enumerate(qa_pairs, 1):
            question = pair["question"]
            reference = pair["reference_answer"]
            
            print(f"\n📝 Q{i}/{len(qa_pairs)}: {question[:60]}...")
            
            try:
                start_time = time.time()
                result = self.rag.ask_question(question, use_compression=False)  
                response_time = time.time() - start_time
                
                answer = result["answer"]
                sources = result.get("sources", [])
                
                predictions.append(answer)
                references.append(reference)
                answer_lengths.append(len(answer))
                response_times.append(response_time)
                
                print(f"   ⏱️  {response_time:.2f}s | 📏 {len(answer)} chars | 📚 {len(sources)} sources")
                
                # Calculate Faithfulness
                if sources:
                    source_content = " ".join([s.get("content", "") for s in sources])
                    answer_words = set(answer.lower().split())
                    source_words = set(source_content.lower().split())

                    # Filter stop words
                    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "and", "or", "of", "by"}
                    answer_words -= stop_words
                    source_words -= stop_words
                    
                    if answer_words:
                        faithfulness = len(answer_words & source_words) / len(answer_words)
                        faithfulness_scores.append(faithfulness)
                        print(f"   ✅ Faithfulness: {faithfulness:.1%}")
                    else:
                        faithfulness_scores.append(0)
                else:
                    faithfulness_scores.append(0)
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
                predictions.append("")
                references.append(reference)

        # Calculate ROUGE scores
        similarity_results = {}
        if ROUGE_AVAILABLE:
            print("\n🚀 Computing ROUGE Scores...")
            rouge_scores = self._calculate_rouge_scores(predictions, references)
            similarity_results.update(rouge_scores)
            print("   ✅ ROUGE Scores computed")
        
        # Summarize results
        results = {
            "avg_faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0,
            "avg_answer_length": np.mean(answer_lengths) if answer_lengths else 0,
            "avg_response_time": np.mean(response_times) if response_times else 0,
            "total_qa_pairs": len(qa_pairs),
            **similarity_results
        }
        
        print("\n" + "-"*60)
        print("📊 ANSWER QUALITY SUMMARY:")
        print(f"   Avg Faithfulness: {results['avg_faithfulness']:.1%}")
        print(f"   Avg Answer Length: {results['avg_answer_length']:.0f} chars")
        print(f"   Avg Response Time: {results['avg_response_time']:.2f}s")
        
        if "rougeL_f1" in results:
            print(f"   ROUGE-1 F1: {results['rouge1_f1']:.3f}")
            print(f"   ROUGE-L F1: {results['rougeL_f1']:.3f}")
        
        print("-"*60)
        
        self.results["answer_quality"] = results
        return results
    
    def _calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate ROUGE scores"""
        if not ROUGE_AVAILABLE or not predictions or not references:
            return {}
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                continue
            
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except:
                continue
        
        return {
            "rouge1_f1": np.mean(rouge1_scores) if rouge1_scores else 0,
            "rouge2_f1": np.mean(rouge2_scores) if rouge2_scores else 0,
            "rougeL_f1": np.mean(rougeL_scores) if rougeL_scores else 0
        }
    
    # ========================================
    # 3. Quick Question Buttons Quality Evaluation
    # ========================================
    
    def evaluate_quick_questions(self, quick_question_tests: Dict[str, List[Dict]]) -> Dict:
        """
        Evaluate quick question button quality
        
        Args:
            quick_question_tests: Quick question test data
                Format: {
                    "tenant": [{"button_label": "...", "question": "...", ...}],
                    "landlord": [{"button_label": "...", "question": "...", ...}]
                }
        
        Returns:
            Quick question quality metrics
        """
        print("\n" + "="*60)
        print("💡 3. QUICK QUESTION BUTTONS QUALITY EVALUATION")
        print("="*60)
        
        if not quick_question_tests:
            print("⚠️ No quick question tests provided")
            return {}
        
        if not isinstance(quick_question_tests, dict):
            print(f"❌ Error: quick_questions must be a dict, got {type(quick_question_tests)}")
            return {}
        
        results_by_role = {}
        
        for role, questions in quick_question_tests.items():
            # Skip metadata keys (e.g., _comment)
            if role.startswith('_'):
                continue
            
            print(f"\n{'='*60}")
            print(f"🏷️  Testing Quick Questions for: {role.upper()}")
            print(f"{'='*60}")
            
            # Ensure questions is a list
            if not isinstance(questions, list):
                print(f"⚠️ Questions for {role} is not a list (type: {type(questions)}), skipping...")
                continue
            
            if not questions:
                print(f"⚠️ No questions for {role}")
                continue
            
            response_times = []
            keyword_matches = []
            faithfulness_scores = []
            rouge_scores_list = []
            
            for i, test in enumerate(questions, 1):
                # Ensure test is a dictionary
                if not isinstance(test, dict):
                    print(f"\n💡 Quick Q{i}/{len(questions)}: Invalid type {type(test)}, skipping...")
                    continue
                
                button_label = test.get("button_label", f"Button {i}")
                question = test.get("question", "")
                reference = test.get("reference_answer", "")
                expected_keywords = test.get("expected_keywords", [])
                importance = test.get("importance", "medium")
                
                if not question:
                    print(f"\n💡 Quick Q{i}/{len(questions)}: {button_label}")
                    print(f"   ⚠️ No question provided, skipping...")
                    continue
                
                print(f"\n💡 Quick Q{i}/{len(questions)}: {button_label}")
                print(f"   Question: {question[:60]}...")
                print(f"   Importance: {importance.upper()}")
                
                try:
                    # Test response time
                    start_time = time.time()
                    result = self.rag.ask_question(question, use_compression=False)
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                    
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    
                    print(f"   ⏱️  Response: {response_time:.2f}s")
                    print(f"   📏 Answer: {len(answer)} chars")
                    
                    # Check keywords
                    if expected_keywords:
                        answer_lower = answer.lower()
                        keywords_found = sum(1 for kw in expected_keywords 
                                        if kw.lower() in answer_lower)
                        keyword_match_rate = keywords_found / len(expected_keywords)
                        keyword_matches.append(keyword_match_rate)
                        print(f"   🔑 Keywords: {keywords_found}/{len(expected_keywords)} ({keyword_match_rate:.1%})")
                    
                    # Calculate Faithfulness
                    if sources:
                        source_content = " ".join([s.get("content", "") for s in sources])
                        answer_words = set(answer.lower().split())
                        source_words = set(source_content.lower().split())
                        
                        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "and", "or", "of", "by"}
                        answer_words -= stop_words
                        source_words -= stop_words
                        
                        if answer_words:
                            faithfulness = len(answer_words & source_words) / len(answer_words)
                            faithfulness_scores.append(faithfulness)
                            print(f"   ✅ Faithfulness: {faithfulness:.1%}")

                    # Calculate ROUGE
                    if reference and ROUGE_AVAILABLE:
                        scores = self.rouge_scorer.score(reference, answer)
                        rouge_scores_list.append(scores['rougeL'].fmeasure)
                        print(f"   📊 ROUGE-L: {scores['rougeL'].fmeasure:.3f}")

                    # Check response time
                    if response_time <= 2.0:
                        print(f"   🚀 Fast response (≤2s)")
                    elif response_time <= 6.0:
                        print(f"   ⚠️ Acceptable response (2-6s)")
                    else:
                        print(f"   ❌ Slow response (>6s)")

                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    response_times.append(0)
                    keyword_matches.append(0)
                    faithfulness_scores.append(0)

            # Summarize results for this role
            if response_times:
                role_results = {
                    "total_questions": len(questions),
                    "avg_response_time": np.mean(response_times) if response_times else 0,
                    "avg_keyword_match": np.mean(keyword_matches) if keyword_matches else 0,
                    "avg_faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0,
                    "fast_response_rate": sum(1 for t in response_times if t <= 2.0) / len(response_times) if response_times else 0,
                }
                
                if rouge_scores_list:
                    role_results["avg_rougeL_f1"] = np.mean(rouge_scores_list)
                
                results_by_role[role] = role_results
                
                print(f"\n{'-'*60}")
                print(f"📊 {role.upper()} QUICK QUESTIONS SUMMARY:")
                print(f"   Avg Response Time: {role_results['avg_response_time']:.2f}s")
                print(f"   Fast Response Rate: {role_results['fast_response_rate']:.1%} (≤2s)")
                print(f"   Keyword Match: {role_results['avg_keyword_match']:.1%}")
                print(f"   Faithfulness: {role_results['avg_faithfulness']:.1%}")
                if "avg_rougeL_f1" in role_results:
                    print(f"   ROUGE-L: {role_results['avg_rougeL_f1']:.3f}")
                print("-"*60)

        # Summarize overall results
        overall_results = {
            "by_role": results_by_role,
            "total_roles_tested": len(results_by_role),
        }

        # Calculate overall averages
        if results_by_role:
            all_response_times = [r["avg_response_time"] for r in results_by_role.values()]
            all_keyword_matches = [r["avg_keyword_match"] for r in results_by_role.values()]
            all_faithfulness = [r["avg_faithfulness"] for r in results_by_role.values()]
            
            overall_results.update({
                "overall_avg_response_time": np.mean(all_response_times),
                "overall_keyword_match": np.mean(all_keyword_matches),
                "overall_faithfulness": np.mean(all_faithfulness),
            })
        
        print(f"\n{'='*60}")
        print("📊 OVERALL QUICK QUESTIONS SUMMARY:")
        if "overall_avg_response_time" in overall_results:
            print(f"   Overall Avg Response: {overall_results['overall_avg_response_time']:.2f}s")
            print(f"   Overall Keyword Match: {overall_results['overall_keyword_match']:.1%}")
            print(f"   Overall Faithfulness: {overall_results['overall_faithfulness']:.1%}")
        print(f"   Roles Tested: {overall_results['total_roles_tested']}")
        print("="*60)
        
        self.results["quick_questions_quality"] = overall_results
        return overall_results
    
    # ========================================
    # 4. Citation Quality Evaluation
    # ========================================
    
    def evaluate_source_citation(self, citation_tests: List[Dict]) -> Dict:
        """
        Evaluate source citation quality
        
        Args:
            citation_tests: Citation tests
                Format: [
                    {
                        "question": "What is the monthly rent?",
                        "expected_source_pages": [1],
                        "critical_pages": [1]
                    }
                ]
        
        Returns:
            Citation quality metrics
        """
        print("\n" + "="*60)
        print("📚 4. SOURCE CITATION QUALITY EVALUATION")
        print("="*60)
        
        if not citation_tests:
            print("⚠️ No citation tests provided")
            return {}
        
        accuracy_scores = []
        completeness_scores = []
        
        for i, test in enumerate(citation_tests, 1):
            question = test["question"]
            expected_pages = set(test.get("expected_source_pages", []))
            critical_pages = set(test.get("critical_pages", []))
            
            print(f"\n📝 Test {i}/{len(citation_tests)}: {question[:60]}...")
            
            try:
                result = self.rag.ask_question(question)
                sources = result.get("sources", [])
                
                if not sources:
                    print(f"   ⚠️ No sources returned")
                    accuracy_scores.append(0)
                    completeness_scores.append(0)
                    continue

                # Extract returned pages
                returned_pages = set()
                for source in sources:
                    page = source.get("page")
                    if page is not None and isinstance(page, int):
                        returned_pages.add(page)
                
                print(f"   📄 Returned: {sorted(returned_pages)}")
                print(f"   🎯 Expected: {sorted(expected_pages)}")

                # 1. Citation Accuracy
                if returned_pages:
                    correct_pages = returned_pages & expected_pages
                    accuracy = len(correct_pages) / len(returned_pages)
                    accuracy_scores.append(accuracy)
                    print(f"   ✅ Accuracy: {accuracy:.1%} ({len(correct_pages)}/{len(returned_pages)})")
                else:
                    accuracy_scores.append(0)

                # 2. Citation Completeness
                if critical_pages:
                    found_critical = returned_pages & critical_pages
                    completeness = len(found_critical) / len(critical_pages)
                    completeness_scores.append(completeness)
                    print(f"   📊 Completeness: {completeness:.1%} ({len(found_critical)}/{len(critical_pages)})")
                else:
                    completeness_scores.append(1.0)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                accuracy_scores.append(0)
                completeness_scores.append(0)

        # Summarize results
        results = {
            "source_accuracy": np.mean(accuracy_scores) if accuracy_scores else 0,
            "source_completeness": np.mean(completeness_scores) if completeness_scores else 0,
            "total_tests": len(citation_tests)
        }
        
        source_score = (
            results["source_accuracy"] * 0.6 +
            results["source_completeness"] * 0.4
        ) * 100
        
        results["source_citation_score"] = source_score
        
        print("\n" + "-"*60)
        print("📊 SOURCE CITATION SUMMARY:")
        print(f"   Accuracy: {results['source_accuracy']:.1%}")
        print(f"   Completeness: {results['source_completeness']:.1%}")
        print(f"   Overall Score: {source_score:.1f}/100")
        print("-"*60)
        
        self.results["source_citation"] = results
        return results
    
    # ========================================
    # 5. Summary Quality Evaluation
    # ========================================
    
    def evaluate_summary_quality(self, summary_tests: List[Dict]) -> Dict:
        """
        Evaluate summary quality
        
        Args:
            summary_tests: Summary tests
                Format: [
                    {
                        "summary_type": "brief",
                        "required_keywords": ["rent", "lease", "deposit"],
                        "min_length": 100,
                        "max_length": 500,
                        "reference_summary": "..."  
                    }
                ]
        
        Returns:
            Summary quality metrics
        """
        print("\n" + "="*60)
        print("📝 5. SUMMARY QUALITY EVALUATION")
        print("="*60)
        
        if not summary_tests:
            print("⚠️ No summary tests provided")
            return {}
        
        keyword_coverage_scores = []
        length_compliance_scores = []
        rouge_scores_all = {"rouge1": [], "rougeL": []}
        generation_times = []
        
        for i, test in enumerate(summary_tests, 1):
            summary_type = test.get("summary_type", "brief")
            required_keywords = test.get("required_keywords", [])
            min_length = test.get("min_length", 0)
            max_length = test.get("max_length", float('inf'))
            reference_summary = test.get("reference_summary", "")
            
            print(f"\n📋 Summary Test {i}/{len(summary_tests)}: Type={summary_type}")
            
            try:
                start_time = time.time()
                generated_summary = self.rag.summarize_contract(summary_type=summary_type)
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                summary_length = len(generated_summary)
                
                print(f"   ⏱️  {generation_time:.2f}s | 📏 {summary_length} chars")
                
                # Check length
                length_valid = min_length <= summary_length <= max_length
                length_compliance_scores.append(1.0 if length_valid else 0.5)
                
                if length_valid:
                    print(f"   ✅ Length OK ({min_length}-{max_length})")
                else:
                    print(f"   ⚠️ Length: {summary_length} (expected {min_length}-{max_length})")
                
                # Check keywords
                keywords_found = sum(1 for kw in required_keywords 
                                    if kw.lower() in generated_summary.lower())
                keyword_coverage = keywords_found / len(required_keywords) if required_keywords else 1.0
                keyword_coverage_scores.append(keyword_coverage)
                
                print(f"   📌 Keywords: {keywords_found}/{len(required_keywords)} ({keyword_coverage:.1%})")
                
                # Calculate ROUGE
                if reference_summary and ROUGE_AVAILABLE:
                    scores = self.rouge_scorer.score(reference_summary, generated_summary)
                    rouge_scores_all["rouge1"].append(scores['rouge1'].fmeasure)
                    rouge_scores_all["rougeL"].append(scores['rougeL'].fmeasure)
                    print(f"   📊 ROUGE-L: {scores['rougeL'].fmeasure:.3f}")
                
            except Exception as e:
                print(f"   ⚠️ Error: {e}")

        # Summarize results
        results = {
            "total_tests": len(summary_tests),
            "avg_keyword_coverage": np.mean(keyword_coverage_scores) if keyword_coverage_scores else 0,
            "length_compliance_rate": np.mean(length_compliance_scores) if length_compliance_scores else 0,
            "avg_generation_time": np.mean(generation_times) if generation_times else 0
        }
        
        if rouge_scores_all["rouge1"]:
            results["avg_rouge1_f1"] = np.mean(rouge_scores_all["rouge1"])
            results["avg_rougeL_f1"] = np.mean(rouge_scores_all["rougeL"])
        
        print("\n" + "-"*60)
        print("📊 SUMMARY QUALITY SUMMARY:")
        print(f"   Keyword Coverage: {results['avg_keyword_coverage']:.1%}")
        print(f"   Length Compliance: {results['length_compliance_rate']:.1%}")
        print(f"   Avg Generation Time: {results['avg_generation_time']:.2f}s")
        if "avg_rougeL_f1" in results:
            print(f"   Avg ROUGE-L: {results['avg_rougeL_f1']:.3f}")
        print("-"*60)
        
        self.results["summary_quality"] = results
        return results
    
    # ========================================
    # 6. Extraction Accuracy Evaluation
    # ========================================
    
    def evaluate_extraction_accuracy(self, ground_truth: Dict) -> Dict:
        """
        Evaluate information extraction accuracy
        
        Args:
            ground_truth: Ground truth dictionary
                Format: {
                    "rent_amount": "s$7,500 per month",
                    "lease_duration": "24 months",
                    ...
                }
        
        Returns:
            Extraction accuracy metrics
        """
        print("\n" + "="*60)
        print("🔍 6. INFORMATION EXTRACTION ACCURACY")
        print("="*60)
        
        if not ground_truth:
            print("⚠️ No ground truth provided")
            return {}
        
        try:
            start_time = time.time()
            extracted = self.rag.extract_key_information()
            extraction_time = time.time() - start_time
            
            print(f"\n⏱️  Extraction time: {extraction_time:.2f}s")
            print(f"📊 Fields: {len(extracted)}")
            
            correct = 0
            partial = 0
            missing = 0
            total = len(ground_truth)
            
            field_results = {}
            
            for field, true_value in ground_truth.items():
                extracted_value = extracted.get(field, "")
                
                # Normalize for comparison
                true_value_norm = self._normalize_for_comparison(true_value)
                extracted_value_norm = self._normalize_for_comparison(extracted_value)
                
                # Exact match
                if true_value_norm == extracted_value_norm:
                    correct += 1
                    match_status = "✅ EXACT"
                    score = 1.0
                # Partial match
                elif true_value_norm in extracted_value_norm or extracted_value_norm in true_value_norm:
                    partial += 1
                    match_status = "⚠️ PARTIAL"
                    score = 0.5
                # Fuzzy match (numerical)
                elif self._fuzzy_match(true_value_norm, extracted_value_norm):
                    partial += 1
                    match_status = "⚠️ FUZZY"
                    score = 0.7
                else:
                    missing += 1
                    match_status = "❌ MISS"
                    score = 0.0
                
                field_results[field] = {
                    "expected": true_value,
                    "extracted": extracted_value,
                    "status": match_status,
                    "score": score
                }
                
                print(f"\n📌 {field}:")
                print(f"   Expected:  {true_value}")
                print(f"   Extracted: {extracted_value}")
                print(f"   {match_status}")

            # Calculate metrics
            precision = (correct + 0.5 * partial) / total if total > 0 else 0
            recall = (correct + 0.5 * partial) / total if total > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results = {
                "exact_match_rate": correct / total if total > 0 else 0,
                "partial_match_rate": partial / total if total > 0 else 0,
                "miss_rate": missing / total if total > 0 else 0,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "correct_fields": correct,
                "partial_fields": partial,
                "missing_fields": missing,
                "total_fields": total,
                "extraction_time": extraction_time,
                "field_details": field_results
            }
            
            print("\n" + "-"*60)
            print("📊 EXTRACTION SUMMARY:")
            print(f"   Exact Match: {results['exact_match_rate']:.1%} ({correct}/{total})")
            print(f"   Partial Match: {results['partial_match_rate']:.1%} ({partial}/{total})")
            print(f"   F1 Score: {results['f1_score']:.3f}")
            print(f"   Extraction Time: {extraction_time:.2f}s")
            print("-"*60)
            
            self.results["extraction_accuracy"] = results
            return results
            
        except Exception as e:
            print(f"⚠️ Extraction error: {e}")
            return {}
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison (consistent with RAG system)"""
        if not isinstance(text, str):
            text = str(text)

        # Apply the same normalization as the RAG system
        text = text.lower().strip()

        # Handle dollar sign
        text = text.replace('$', 's$')

        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFKD', text)

        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """Fuzzy match (numerical)"""
        import re
        nums1 = re.findall(r'\d+', str1)
        nums2 = re.findall(r'\d+', str2)
        return len(nums1) > 0 and nums1 == nums2
    
    # ========================================
    # 7. Efficiency Evaluation
    # ========================================
    
    def evaluate_efficiency(self, questions: List[str], runs: int = 3) -> Dict:
        """
        Evaluate system efficiency
        
        Args:
            questions: List of test questions
            runs: Number of repetitions for each question

        Returns:
            System efficiency metrics
        """

        print("\n" + "="*60)
        print("⚡ 7. SYSTEM EFFICIENCY EVALUATION")
        print("="*60)
        
        if not questions:
            print("⚠️ No questions provided")
            return {}
        
        response_times = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Q{i}/{len(questions)}: {question[:50]}...")
            
            question_times = []
            for run in range(runs):
                try:
                    start_time = time.time()
                    self.rag.ask_question(question)
                    elapsed = time.time() - start_time
                    question_times.append(elapsed)
                except Exception as e:
                    print(f"   ⚠️ Run {run+1} error: {e}")
            
            if question_times:
                avg_time = np.mean(question_times)
                response_times.append(avg_time)
                print(f"   ⏱️  Avg: {avg_time:.2f}s (over {runs} runs)")
        
        results = {
            "avg_response_time": np.mean(response_times) if response_times else 0,
            "min_response_time": np.min(response_times) if response_times else 0,
            "max_response_time": np.max(response_times) if response_times else 0,
            "std_response_time": np.std(response_times) if response_times else 0,
            "total_questions": len(questions),
            "runs_per_question": runs
        }
        
        print("\n" + "-"*60)
        print("📊 EFFICIENCY SUMMARY:")
        print(f"   Avg Response: {results['avg_response_time']:.2f}s")
        print(f"   Min/Max: {results['min_response_time']:.2f}s / {results['max_response_time']:.2f}s")
        print(f"   Std Dev: {results['std_response_time']:.2f}s")
        print("-"*60)
        
        self.results["efficiency"] = results
        return results
    
    # ========================================
    # Full Evaluation Runner
    # ========================================
    
    def run_full_evaluation(self, test_data: Dict) -> Dict:
        """Run full evaluation"""
        print("\n" + "="*80)
        print(" "*20 + "🚀 RENTALPEACE CHATBOT EVALUATION")
        print(" "*25 + f"Mode: {self.evaluation_mode.upper()}")
        print(" "*22 + f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        start_time = time.time()
        
        # 1. Retrieval Quality
        if "retrieval_tests" in test_data and test_data["retrieval_tests"]:
            self.evaluate_retrieval_quality(test_data["retrieval_tests"])
        
        # 2. Answer Quality
        if "qa_pairs" in test_data and test_data["qa_pairs"]:
            self.evaluate_answer_quality(test_data["qa_pairs"])
        
        # 3. Quick Question Buttons Quality
        if "quick_questions" in test_data and test_data["quick_questions"]:
            self.evaluate_quick_questions(test_data["quick_questions"])
        
        # 4. Source Citation
        if "source_citation_tests" in test_data and test_data["source_citation_tests"]:
            self.evaluate_source_citation(test_data["source_citation_tests"])
        
        # 5. Summary Quality
        if "summary_tests" in test_data and test_data["summary_tests"]:
            self.evaluate_summary_quality(test_data["summary_tests"])
        
        # 6. Information Extraction
        if "extraction_ground_truth" in test_data and test_data["extraction_ground_truth"]:
            self.evaluate_extraction_accuracy(test_data["extraction_ground_truth"])
        
        # 7. Efficiency
        if "efficiency_questions" in test_data and test_data["efficiency_questions"]:
            self.evaluate_efficiency(test_data["efficiency_questions"])
        
        total_time = time.time() - start_time
        self.results["total_evaluation_time"] = total_time

        # Calculate overall score
        self._calculate_overall_score()

        # Print summary
        self.print_summary(total_time)
        
        return self.results
    
    def _calculate_overall_score(self):
        """Calculate overall score"""
        weights = {
            "retrieval_quality": 0.20,
            "answer_quality": 0.25,
            "quick_questions_quality": 0.15,  
            "source_citation": 0.12,
            "summary_quality": 0.12,
            "extraction_accuracy": 0.10,
            "efficiency": 0.06
        }
        
        dimension_scores = {}
        
        # Retrieval Quality
        if self.results.get("retrieval_quality"):
            r = self.results["retrieval_quality"]
            dimension_scores["retrieval_quality"] = (r.get("mrr", 0) * 50 + r.get("recall@5", 0) * 50)
        
        # Answer Quality
        if self.results.get("answer_quality"):
            a = self.results["answer_quality"]
            dimension_scores["answer_quality"] = (a.get("avg_faithfulness", 0) * 60 + 
                                                  a.get("rougeL_f1", 0) * 40)

        # Quick Question Buttons Quality
        if self.results.get("quick_questions_quality"):
            q = self.results["quick_questions_quality"]
            if "overall_avg_response_time" in q:
                # Score: Response Time + Keyword Match + Faithfulness
                speed_score = max(0, 100 - max(0, q["overall_avg_response_time"]-2) * 20)  # Full score within 2 seconds
                keyword_score = q["overall_keyword_match"] * 100
                faithfulness_score = q["overall_faithfulness"] * 100
                dimension_scores["quick_questions_quality"] = (
                    speed_score * 0.2 + keyword_score * 0.4 + faithfulness_score * 0.4
                )
            else:
                dimension_scores["quick_questions_quality"] = 0

        # Source Citation
        if self.results.get("source_citation"):
            dimension_scores["source_citation"] = self.results["source_citation"].get("source_citation_score", 0)

        # Summary Quality
        if self.results.get("summary_quality"):
            s = self.results["summary_quality"]
            dimension_scores["summary_quality"] = (s.get("avg_keyword_coverage", 0) * 70 + 
                                                   s.get("avg_rougeL_f1", 0) * 30)

        # Information Extraction
        if self.results.get("extraction_accuracy"):
            dimension_scores["extraction_accuracy"] = self.results["extraction_accuracy"].get("f1_score", 0) * 100

        # Efficiency
        if self.results.get("efficiency"):
            e = self.results["efficiency"]
            response_time = e.get("avg_response_time", 10)
            dimension_scores["efficiency"] = max(0, 100 - max(0, (response_time-3) * 10))

        # Calculate weighted total score
        total_score = 0
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0)
            total_score += score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        self.results["dimension_scores"] = dimension_scores
        self.results["overall_score"] = overall_score
    
    def print_summary(self, total_time: float = 0):
        """Print evaluation summary"""
        print("\n\n" + "="*80)
        print(" "*30 + "📊 EVALUATION SUMMARY")
        print("="*80)

        # Retrieval Quality
        if self.results.get("retrieval_quality"):
            r = self.results["retrieval_quality"]
            print(f"\n🎯 Retrieval Quality:")
            print(f"   • MRR: {r.get('mrr', 0):.3f}")
            print(f"   • Recall@5: {r.get('recall@5', 0):.1%}")

        # Answer Quality
        if self.results.get("answer_quality"):
            a = self.results["answer_quality"]
            print(f"\n💬 Answer Quality:")
            print(f"   • Faithfulness: {a.get('avg_faithfulness', 0):.1%}")
            if a.get('rougeL_f1'):
                print(f"   • ROUGE-L: {a.get('rougeL_f1', 0):.3f}")

        # Quick Questions Quality
        if self.results.get("quick_questions_quality"):
            q = self.results["quick_questions_quality"]
            print(f"\n💡 Quick Questions Quality:")
            if "overall_avg_response_time" in q:
                print(f"   • Avg Response: {q['overall_avg_response_time']:.2f}s")
                print(f"   • Keyword Match: {q['overall_keyword_match']:.1%}")
                print(f"   • Faithfulness: {q['overall_faithfulness']:.1%}")
            
            if "by_role" in q:
                for role, stats in q["by_role"].items():
                    print(f"   • {role.capitalize()}: {stats['avg_response_time']:.2f}s avg")

        # Source Citation
        if self.results.get("source_citation"):
            sc = self.results["source_citation"]
            print(f"\n📚 Source Citation:")
            print(f"   • Accuracy: {sc.get('source_accuracy', 0):.1%}")
            print(f"   • Score: {sc.get('source_citation_score', 0):.1f}/100")

        # Summary Quality
        if self.results.get("summary_quality"):
            s = self.results["summary_quality"]
            print(f"\n📝 Summary Quality:")
            print(f"   • Keyword Coverage: {s.get('avg_keyword_coverage', 0):.1%}")
            print(f"   • Generation Time: {s.get('avg_generation_time', 0):.2f}s")

        # Information Extraction
        if self.results.get("extraction_accuracy"):
            ex = self.results["extraction_accuracy"]
            print(f"\n🔍 Extraction Accuracy:")
            print(f"   • F1 Score: {ex.get('f1_score', 0):.3f}")
            print(f"   • Exact Match: {ex.get('exact_match_rate', 0):.1%}")

        # Efficiency
        if self.results.get("efficiency"):
            e = self.results["efficiency"]
            print(f"\n⚡ Efficiency:")
            print(f"   • Avg Response: {e.get('avg_response_time', 0):.2f}s")

        # Dimension Scores Visualization
        if "dimension_scores" in self.results:
            print(f"\n{'='*80}")
            print(" "*25 + "📈 DIMENSION SCORES (0-100)")
            print("="*80)
            
            dimension_scores = self.results["dimension_scores"]
            dimension_names = {
                "retrieval_quality": "🎯 Retrieval Quality",
                "answer_quality": "💬 Answer Quality",
                "quick_questions_quality": "💡 Quick Questions Quality",
                "source_citation": "📚 Source Citation",
                "summary_quality": "📝 Summary Quality",
                "extraction_accuracy": "🔍 Extraction Accuracy",
                "efficiency": "⚡ System Efficiency"
            }
            
            
            ordered_dimensions = [
                "retrieval_quality",
                "answer_quality",
                "quick_questions_quality",
                "source_citation",
                "summary_quality",
                "extraction_accuracy",
                "efficiency"
            ]
            
            for dim in ordered_dimensions:
                if dim in dimension_scores:
                    score = dimension_scores[dim]
                    name = dimension_names.get(dim, dim)
                    
                    # Visual bar
                    bar_length = 40
                    filled = int(bar_length * score / 100)
                    bar = "█" * filled + "░" * (bar_length - filled)

                    # color indicator
                    if score >= 80:
                        indicator = "🟢"
                    elif score >= 60:
                        indicator = "🟡"
                    else:
                        indicator = "🔴"
                    
                    print(f"{indicator} {name:30s} [{bar}] {score:5.1f}/100")
            
            print("="*80)


        # Overall Score
        if "overall_score" in self.results:
            print(f"\n{'='*80}")
            print(f"   🏆 OVERALL SCORE: {self.results['overall_score']:.1f}/100")
            
            score = self.results['overall_score']
            if score >= 88:
                rating = "⭐⭐⭐⭐⭐ Excellent"
            elif score >= 73:
                rating = "⭐⭐⭐⭐ Good"
            elif score >= 58:
                rating = "⭐⭐⭐ Fair"
            else:
                rating = "⭐⭐ Needs Improvement"
            
            print(f"   Rating: {rating}")
            print(f"{'='*80}")
        
        if total_time > 0:
            print(f"\n⏱️  Total Time: {total_time:.2f}s")

        self.print_score_breakdown_tree()
        
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """Save evaluation results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {output_path}")

    def print_score_breakdown_tree(self):
        """
        Print score breakdown in a tree structure
        """
        print("\n" + "="*80)
        print(" "*20 + "🌳 SCORE CALCULATION BREAKDOWN")
        print(" "*25 + "(Tree Structure)")
        print("="*80)
        
        if "overall_score" not in self.results:
            print("⚠️ Overall score not calculated yet")
            return
        
        overall_score = self.results["overall_score"]
        dimension_scores = self.results.get("dimension_scores", {})
        
        # Define dimension weights (consistent with _calculate_overall_score)
        weights = {
            "retrieval_quality": 0.20,
            "answer_quality": 0.25,
            "quick_questions_quality": 0.15,
            "source_citation": 0.12,
            "summary_quality": 0.12,
            "extraction_accuracy": 0.10,
            "efficiency": 0.06
        }

        # Dimension name mapping
        dimension_names = {
            "retrieval_quality": "🎯 Retrieval Quality",
            "answer_quality": "💬 Answer Quality",
            "quick_questions_quality": "💡 Quick Questions Quality",
            "source_citation": "📚 Source Citation",
            "summary_quality": "📝 Summary Quality",
            "extraction_accuracy": "🔍 Extraction Accuracy",
            "efficiency": "⚡ System Efficiency"
        }

        # Print root node
        print(f"\n🏆 Overall Score: {overall_score:.1f}/100")
        print("│")

        # Sort by weight (from high to low)
        sorted_dimensions = sorted(
            dimension_scores.keys(),
            key=lambda x: weights.get(x, 0),
            reverse=True
        )
        
        for idx, dimension in enumerate(sorted_dimensions):
            score = dimension_scores[dimension]
            weight = weights.get(dimension, 0)
            name = dimension_names.get(dimension, dimension)

            # Check if it's the last dimension
            is_last = (idx == len(sorted_dimensions) - 1)
            prefix = "└─" if is_last else "├─"
            extension = "  " if is_last else "│ "

            # Print dimension node
            print(f"{prefix} [{weight:.0%}] {name} ({score:.1f}/100)")

            # Print sub-metrics for this dimension
            self._print_dimension_details(dimension, extension)
        
        print("\n" + "="*80)
        print("Legend: [X%] = Weight in overall score calculation")
        print("="*80 + "\n")


    def _print_dimension_details(self, dimension: str, prefix: str):
        """
        Print detailed calculation methods and sub-metrics for each dimension

        Args:
            dimension: Dimension name
            prefix: Indentation prefix
        """
        # 1. Retrieval Quality
        if dimension == "retrieval_quality" and self.results.get("retrieval_quality"):
            r = self.results["retrieval_quality"]
            print(f"{prefix}   ├─ [50%] MRR: {r.get('mrr', 0):.3f}")
            print(f"{prefix}   └─ [50%] Recall@5: {r.get('recall@5', 0):.1%}")
        
        # 2. Answer Quality
        elif dimension == "answer_quality" and self.results.get("answer_quality"):
            a = self.results["answer_quality"]
            print(f"{prefix}   ├─ [60%] Faithfulness: {a.get('avg_faithfulness', 0):.1%}")
            print(f"{prefix}   └─ [40%] ROUGE-L F1: {a.get('rougeL_f1', 0):.3f}")
        
        # 3. Quick Questions Quality
        elif dimension == "quick_questions_quality" and self.results.get("quick_questions_quality"):
            q = self.results["quick_questions_quality"]
            if "overall_avg_response_time" in q:
                # Recalculate sub-scores (consistent with _calculate_overall_score)
                speed_score = max(0, 100 - max(0, q["overall_avg_response_time"]-2) * 20)
                keyword_score = q["overall_keyword_match"] * 100
                faithfulness_score = q["overall_faithfulness"] * 100
                
                print(f"{prefix}   ├─ [20%] Speed Score: {speed_score:.1f}/100")
                print(f"{prefix}   │      (Avg Response: {q['overall_avg_response_time']:.2f}s, target: ≤2s)")
                print(f"{prefix}   ├─ [40%] Keyword Match: {q['overall_keyword_match']:.1%}")
                print(f"{prefix}   └─ [40%] Faithfulness: {q['overall_faithfulness']:.1%}")
        
        # 4. Source Citation
        elif dimension == "source_citation" and self.results.get("source_citation"):
            sc = self.results["source_citation"]
            print(f"{prefix}   ├─ [60%] Source Accuracy: {sc.get('source_accuracy', 0):.1%}")
            print(f"{prefix}   └─ [40%] Source Completeness: {sc.get('source_completeness', 0):.1%}")
        
        # 5. Summary Quality
        elif dimension == "summary_quality" and self.results.get("summary_quality"):
            s = self.results["summary_quality"]
            print(f"{prefix}   ├─ [70%] Keyword Coverage: {s.get('avg_keyword_coverage', 0):.1%}")
            print(f"{prefix}   └─ [30%] ROUGE-L F1: {s.get('avg_rougeL_f1', 0):.3f}")
        
        # 6. Extraction Accuracy
        elif dimension == "extraction_accuracy" and self.results.get("extraction_accuracy"):
            ex = self.results["extraction_accuracy"]
            print(f"{prefix}   └─ [100%] F1 Score: {ex.get('f1_score', 0):.3f}")
            print(f"{prefix}        (Exact: {ex.get('exact_match_rate', 0):.1%}, " +
                f"Partial: {ex.get('partial_match_rate', 0):.1%})")
        
        # 7. Efficiency
        elif dimension == "efficiency" and self.results.get("efficiency"):
            e = self.results["efficiency"]
            response_time = e.get("avg_response_time", 0)
            print(f"{prefix}   └─ [100%] Response Time: {response_time:.2f}s (target: ≤3s)")
            print(f"{prefix}        Formula: max(0, 100 - max(0, (time-3)*10))")

# ========================================
# Main Program
# ========================================

def main():
    """Main function"""
    API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    TEST_DATA_FILE = "test_data_rentalpeace.json"
    CONTRACT_PDF = "Track_B_Tenancy_Agreement.pdf"
    
    print("\n" + "="*80)
    print(" "*20 + "🤖 RENTALPEACE EVALUATOR")
    print("="*80)
    
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Initialize RAG system
    print(f"\n📝 Initializing RAG system...")
    rag = AdvancedContractRAG(api_key=API_KEY, model=MODEL)

    # Load contract
    if Path(CONTRACT_PDF).exists():
        print(f"📄 Loading contract: {CONTRACT_PDF}")
        result = rag.load_pdf(CONTRACT_PDF)
        if result.get("success"):
            print(f"✅ Contract loaded: {result['stats']}")
        else:
            print(f"❌ Failed: {result.get('error')}")
            return
    else:
        print(f"⚠️ Contract not found: {CONTRACT_PDF}")
        return

    # Load test data
    if Path(TEST_DATA_FILE).exists():
        print(f"📂 Loading test data: {TEST_DATA_FILE}")
        with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        print(f"⚠️ Test data not found: {TEST_DATA_FILE}")
        return

    # Create evaluator
    evaluator = RentalPeaceEvaluator(rag_system=rag, evaluation_mode="fast")

    # Run evaluation
    print("\n🚀 Starting evaluation...\n")
    results = evaluator.run_full_evaluation(test_data)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    evaluator.save_results(output_file)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Results: {output_file}")


if __name__ == "__main__":
    main()