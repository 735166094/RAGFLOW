# rag_paper_assistant/assistant.py
"""
统一入口类 - 封装所有功能
"""

from typing import List, Dict, Any, Optional
import os
import json

from parser.enhanced_parser import EnhancedPDFParser
from chunking.hierarchical_chunker import HierarchicalChunker
from retrieval.hybrid_retriever import HybridRetriever
from generation.prompt_templates import PaperPromptTemplate
from generation.hallucination_detector import HallucinationDetector


class PaperAssistant:
    """
    论文助手统一入口类
    
    使用示例：
    >>> assistant = PaperAssistant(mode="method")
    >>> assistant.load_paper("paper.pdf")
    >>> result = assistant.query("Deformable DETR的attention怎么实现？")
    """
    
    def __init__(self, 
                 mode: str = "method",
                 embedding_model: str = "Dmeta-embedding-zh",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cuda"):
        
        self.mode = mode
        self.device = device
        
        # 初始化各组件
        self.parser = EnhancedPDFParser(use_ocr=True, use_latex=True)
        self.chunker = HierarchicalChunker()
        self.retriever = HybridRetriever(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            device=device
        )
        self.prompt_template = PaperPromptTemplate(mode=mode)
        self.hallucination_detector = HallucinationDetector()
        
        # 状态
        self.papers_loaded = []
        self.is_index_built = False
        
    def load_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        加载并解析单篇论文
        """
        print(f"正在解析: {pdf_path}")
        
        # 1. 解析PDF
        parsed = self.parser.parse_paper(pdf_path)
        
        # 2. 切片
        chunks = self.chunker.chunk_paper(parsed)
        
        # 3. 保存元数据
        paper_info = {
            "path": pdf_path,
            "title": parsed["metadata"].get("title", os.path.basename(pdf_path)),
            "chunks": chunks,
            "chunk_count": len(chunks)
        }
        self.papers_loaded.append(paper_info)
        
        print(f"解析完成: {paper_info['title']}, 生成 {len(chunks)} 个切片")
        return paper_info
    
    def load_papers(self, pdf_paths: List[str]):
        """批量加载论文"""
        for path in pdf_paths:
            self.load_paper(path)
        
        # 构建检索索引
        self._build_index()
        
    def _build_index(self):
        """构建检索索引"""
        all_chunks = []
        for paper in self.papers_loaded:
            all_chunks.extend(paper["chunks"])
        
        print(f"构建索引: {len(all_chunks)} 个切片")
        self.retriever.build_index(all_chunks)
        self.is_index_built = True
        
    def query(self, 
              query: str,
              top_k: int = 5,
              return_sources: bool = True) -> Dict[str, Any]:
        """
        查询接口
        """
        if not self.is_index_built:
            raise RuntimeError("请先加载论文并构建索引")
        
        # 1. 检索
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        
        # 2. 构建上下文
        context = "\n\n".join([chunk.text for chunk, score in retrieved[:3]])
        
        # 3. 生成Prompt
        prompt = self.prompt_template.format(query, context)
        
        # 4. 调用Ollama生成（这里简化，实际需要HTTP调用）
        # answer = self._call_ollama(prompt)
        answer = f"[模拟回答] 基于检索结果生成的答案: {context[:100]}..."
        
        # 5. 幻觉检测
        confidence = self.hallucination_detector.compute_confidence(answer, context)
        answer = self.hallucination_detector.mark_low_confidence(answer, confidence)
        
        # 6. 组装结果
        result = {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "mode": self.mode,
            "sources": [
                {
                    "text": chunk.text[:200] + "...",
                    "metadata": chunk.metadata,
                    "score": float(score)
                }
                for chunk, score in retrieved[:top_k]
            ] if return_sources else None
        }
        
        return result
    
    def switch_mode(self, mode: str):
        """切换助手模式"""
        self.mode = mode
        self.prompt_template.switch_mode(mode)
        print(f"已切换到 {mode} 模式")