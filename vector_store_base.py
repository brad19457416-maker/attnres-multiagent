"""
💾 向量存储抽象基类
===

模块化抽象：支持不同向量数据库插拔替换

当前支持：
- 内置内存向量存储（简单场景）
- ChromaDB（推荐，本地持久化）
- 可扩展其他：Pinecone, Weaviate, Qdrant 等
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import math


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add(self, embedding: List[float], metadata: dict) -> str:
        """添加一个向量，返回id"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """搜索相似向量，返回 (id, similarity, metadata) 列表"""
        pass
    
    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """删除向量"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """返回存储的向量数量"""
        pass


class InMemoryVectorStore(VectorStore):
    """简单内存向量存储 - 适用于小数据量测试
    
    直接余弦相似度计算，不需要外部依赖
    """
    
    def __init__(self):
        self.vectors: List[dict] = []  # [{id, embedding, metadata}, ...]
        self.next_id = 1
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def add(self, embedding: List[float], metadata: dict) -> str:
        vector_id = f"vec_{self.next_id}"
        self.next_id += 1
        self.vectors.append({
            "id": vector_id,
            "embedding": embedding,
            "metadata": metadata,
        })
        return vector_id
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, dict]]:
        results: List[Tuple[float, str, dict]] = []
        
        for vec in self.vectors:
            sim = self._cosine_similarity(query_embedding, vec["embedding"])
            results.append((-sim, vec["id"], vec["metadata"]))  # 负号用于升序排序
        
        # 排序：相似度降序
        results.sort()
        # 取 top-k，恢复正相似度
        top_results = [(-sim, id_, meta) for sim, id_, meta in results[:top_k]]
        return top_results
    
    def delete(self, vector_id: str) -> bool:
        for i, vec in enumerate(self.vectors):
            if vec["id"] == vector_id:
                del self.vectors[i]
                return True
        return False
    
    def count(self) -> int:
        return len(self.vectors)


class ChromaDBVectorStore(VectorStore):
    """ChromaDB 向量存储 - 推荐用于生产
    
    需要安装: pip install chromadb
    """
    
    def __init__(self, collection_name: str = "default", persist_dir: str = "./chroma_db"):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
    
    def add(self, embedding: List[float], metadata: dict) -> str:
        vector_id = f"vec_{self.collection.count() + 1}"
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[vector_id],
        )
        return vector_id
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        # 解析结果
        output: List[Tuple[str, float, dict]] = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, vec_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                # chroma 使用 L2 距离，转换为相似度
                # similarity = 1 / (1 + distance)
                similarity = 1 / (1 + distance)
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                output.append((vec_id, similarity, metadata))
        return output
    
    def delete(self, vector_id: str) -> bool:
        try:
            self.collection.delete(ids=[vector_id])
            return True
        except Exception:
            return False
    
    def count(self) -> int:
        return self.collection.count()


# 默认实例
default_in_memory = InMemoryVectorStore()
