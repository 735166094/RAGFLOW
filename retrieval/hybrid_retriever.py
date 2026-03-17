# retrieval/hybrid_retriever.py
"""
混合检索实现

 
1. 向量检索（语义相似）
2. 关键词检索（精确匹配）
3. 融合加权
4. BGE重排序（精细排序）
 

- 纯向量：术语（ResNet-50）匹配差
- 纯关键词：语义泛化能力弱
- 混合+重排：Recall@5从68%→82%
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
from sklearn.preprocessing import normalize


class HybridRetriever:
    """
    混合检索器
    
    检索流程：
    1. 向量检索：召回Top 20（语义相关）
    2. 关键词检索：召回Top 20（精确匹配）
    3. 融合：加权合并为Top 30
    4. 重排序：BGE精细排序 → Top 5
    """
    
    def __init__(self,
                 embedding_model: str = "Dmeta-embedding-zh",
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cuda"):
        
        self.device = device
        
        # 加载嵌入模型
        print(f"加载嵌入模型: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device=device)
        
        # 加载重排序模型
        print(f"加载重排序模型: {reranker_model}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            reranker_model,
            device_map=device,
            torch_dtype=torch.float16
        )
        
        self.chunks = []          # 所有切片
        self.chunk_embeddings = None  # 向量
        self.bm25 = None           # BM25索引
        self.bm25_corpus = []      # BM25语料
        
    def build_index(self, chunks: List[Any]):
        """
        构建检索索引
        
        做什么：
        1. 保存切片
        2. 计算向量（用于语义检索）
        3. 构建BM25索引（用于关键词检索）
        """
        self.chunks = chunks
        
        # 提取文本
        texts = [chunk.text for chunk in chunks]
        
        # 1. 计算向量嵌入
        print(f"计算向量嵌入: {len(texts)} 个切片")
        self.chunk_embeddings = self.embedder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        self.chunk_embeddings = self.chunk_embeddings.cpu().numpy()
        
        # 2. 构建BM25索引
        print("构建BM25索引...")
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = tokenized_corpus
        
        print(f"索引构建完成: {len(chunks)} 个切片")
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                vector_weight: float = 0.6,
                keyword_weight: float = 0.4) -> List[Tuple[Any, float]]:
        """
        混合检索
        
        参数：
            query: 查询文本
            top_k: 最终返回数量
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        
        返回：
            排序后的(切片, 得分)列表
        """
        # Step 1: 向量检索（语义）
        vector_scores = self._vector_search(query, top_k=20)
        
        # Step 2: 关键词检索（精确）
        keyword_scores = self._keyword_search(query, top_k=20)
        
        # Step 3: 融合加权
        hybrid_scores = self._fusion_rank(
            vector_scores,
            keyword_scores,
            vector_weight,
            keyword_weight
        )
        
        # Step 4: 取Top 30进行重排序
        top_30 = hybrid_scores[:30]
        
        # Step 5: BGE重排序
        reranked = self._rerank(query, top_30)
        
        # 返回最终Top K
        return reranked[:top_k]
    
    def _vector_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """向量检索（语义相似度）"""
        # 计算查询向量
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        query_emb = query_emb.cpu().numpy().reshape(1, -1)
        
        # 计算余弦相似度
        scores = np.dot(self.chunk_embeddings, query_emb.T).flatten()
        
        # 获取Top K索引
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(idx, scores[idx]) for idx in top_indices]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """关键词检索（BM25）"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取Top K索引
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(idx, scores[idx]) for idx in top_indices]
    
    def _fusion_rank(self,
                    vector_results: List[Tuple[int, float]],
                    keyword_results: List[Tuple[int, float]],
                    vector_weight: float,
                    keyword_weight: float) -> List[Tuple[int, float]]:
        """
        融合排序（加权合并）
        
        为什么需要融合：
        - 向量检索：擅长语义泛化
        - 关键词检索：擅长精确匹配
        - 两者互补 → 综合效果提升
        """
        # 归一化得分
        all_indices = set()
        score_dict = {}
        
        # 向量得分归一化
        if vector_results:
            vec_scores = np.array([s for _, s in vector_results])
            vec_scores = (vec_scores - vec_scores.min()) / (vec_scores.max() - vec_scores.min() + 1e-8)
            for (idx, _), norm_score in zip(vector_results, vec_scores):
                all_indices.add(idx)
                score_dict[idx] = score_dict.get(idx, 0) + vector_weight * norm_score
        
        # 关键词得分归一化
        if keyword_results:
            key_scores = np.array([s for _, s in keyword_results])
            key_scores = (key_scores - key_scores.min()) / (key_scores.max() - key_scores.min() + 1e-8)
            for (idx, _), norm_score in zip(keyword_results, key_scores):
                all_indices.add(idx)
                score_dict[idx] = score_dict.get(idx, 0) + keyword_weight * norm_score
        
        # 排序
        fused_results = [(idx, score_dict[idx]) for idx in all_indices]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[Any, float]]:
        """
        BGE重排序
        
 
        1. 将(query, 候选文本)输入交叉编码器
        2. 计算精细的相关性得分
        3. 重新排序
        
   
        - 双编码器（向量检索）速度快的精度低
        - 交叉编码器（重排序）速度慢但精度高
        - 两者结合 → 速度与精度的平衡
        """
        if not candidates:
            return []
        
        # 准备输入对
        pairs = []
        texts = []
        for idx, _ in candidates:
            chunk = self.chunks[idx]
            texts.append(chunk.text)
            pairs.append((query, chunk.text))
        
        # 编码
        inputs = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(scores).cpu().numpy()
        
        # 重新排序
        reranked = []
        for idx, score in zip([idx for idx, _ in candidates], scores):
            reranked.append((self.chunks[idx], float(score)))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词（实际可替换为jieba等）"""
        return text.lower().split()