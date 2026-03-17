# api/main.py
"""
FastAPI服务

1. 提供RESTful API
2. 集成Redis缓存
3. 限流保护Ollama
4. Prometheus监控
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import aioredis
from typing import Optional, List, Dict, Any
import time
import hashlib
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import aiohttp
from functools import wraps
import logging

from retrieval.hybrid_retriever import HybridRetriever
from generation.prompt_templates import PaperPromptTemplate
from generation.hallucination_detector import HallucinationDetector

app = FastAPI(title="RAG Paper Assistant")

# 监控指标
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["endpoint"])
RETRIEVAL_LATENCY = Histogram("retrieval_duration_seconds", "Retrieval latency")
GENERATION_LATENCY = Histogram("generation_duration_seconds", "Generation latency")
CACHE_HITS = Counter("cache_hits_total", "Total cache hits")
CACHE_MISSES = Counter("cache_misses_total", "Total cache misses")
OLLAMA_QUEUE_SIZE = Gauge("ollama_queue_size", "Ollama request queue size")

# 请求模型
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_cache: Optional[bool] = True
    stream: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    cached: bool
    latency_ms: int

# 全局组件
retriever = None
prompt_template = PaperPromptTemplate()
hallucination_detector = HallucinationDetector()
redis_client = None
ollama_semaphore = asyncio.Semaphore(5)  # 限流：最多5个并发Ollama请求


@app.on_event("startup")
async def startup_event():
    """服务启动初始化"""
    global retriever, redis_client
    
    # 1. 初始化检索器
    retriever = HybridRetriever()
    # 加载预构建的索引
    # retriever.load_index("data/index")
    
    # 2. 连接Redis
    redis_client = await aioredis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    
    print("服务启动完成")


def cache_key(query: str, top_k: int) -> str:
    """生成缓存键（语义哈希）"""
    # 归一化查询（去除标点、空格）
    normalized = " ".join(query.lower().split())
    # 取前50个字符的哈希
    query_hash = hashlib.md5(normalized[:50].encode()).hexdigest()
    return f"rag:q:{query_hash}:k:{top_k}"


def monitor(endpoint: str):
    """监控装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(method="POST", endpoint=endpoint).inc()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        return wrapper
    return decorator


@app.post("/api/query", response_model=QueryResponse)
@monitor(endpoint="/api/query")
async def query(request: QueryRequest, req: Request):
    """
    核心查询接口
    
    流程：
    1. 检查缓存（命中直接返回）
    2. 混合检索
    3. 生成回答
    4. 幻觉检测
    5. 异步写入缓存
    """
    
    # 1. 检查缓存
    if request.use_cache:
        cache_key_str = cache_key(request.query, request.top_k)
        cached_result = await redis_client.get(cache_key_str)
        
        if cached_result:
            CACHE_HITS.inc()
            data = json.loads(cached_result)
            data["cached"] = True
            return JSONResponse(content=data)
        else:
            CACHE_MISSES.inc()
    
    # 2. 检索阶段（计时）
    start_retrieval = time.time()
    with RETRIEVAL_LATENCY.time():
        retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k * 2)
        sources = [
            {
                "text": chunk.text[:200] + "...",
                "metadata": chunk.metadata,
                "score": float(score)
            }
            for chunk, score in retrieved_chunks[:request.top_k]
        ]
    retrieval_time = (time.time() - start_retrieval) * 1000
    
    # 3. 生成回答（带限流）
    async def generate_with_ollama(query: str, context: str):
        """带限流的Ollama调用"""
        async with ollama_semaphore:
            OLLAMA_QUEUE_SIZE.set(ollama_semaphore._value)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "qwen2:7b",
                    "prompt": prompt_template.format(query, context),
                    "stream": request.stream,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                }
                
                async with session.post("http://localhost:11434/api/generate", 
                                      json=payload) as resp:
                    if request.stream:
                        return StreamingResponse(
                            resp.content,
                            media_type="text/event-stream"
                        )
                    else:
                        result = await resp.json()
                        return result.get("response", "")
    
    # 构建上下文
    context = "\n\n".join([chunk.text for chunk, _ in retrieved_chunks[:3]])
    
    # 生成
    start_generation = time.time()
    with GENERATION_LATENCY.time():
        try:
            answer = await generate_with_ollama(request.query, context)
        except Exception as e:
            # 熔断降级：返回检索结果
            answer = f"[系统降级] 生成服务暂时不可用。以下是相关文献片段：\n\n{context[:500]}..."
            logging.error(f"Ollama调用失败: {e}")
    generation_time = (time.time() - start_generation) * 1000
    
    # 4. 幻觉检测
    confidence = hallucination_detector.compute_confidence(answer, context)
    
    # 5. 构建响应
    response = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "cached": False,
        "latency_ms": int(retrieval_time + generation_time),
        "timing": {
            "retrieval_ms": int(retrieval_time),
            "generation_ms": int(generation_time)
        }
    }
    
    # 6. 异步写入缓存
    if request.use_cache and confidence > 0.8:  # 只缓存高置信度结果
        asyncio.create_task(
            redis_client.setex(
                cache_key_str,
                3600,  # 1小时过期
                json.dumps(response, ensure_ascii=False)
            )
        )
    
    return JSONResponse(content=response)


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/api/metrics")
async def metrics():
    """Prometheus监控指标"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存（用于测试）"""
    await redis_client.flushdb()
    return {"status": "cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)