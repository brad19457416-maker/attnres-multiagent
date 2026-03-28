"""
向量相似度预筛选 - 筛选top-k最相关子结果给聚合器
减少token用量，提升效率
"""

from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class ScoredResult:
    task_id: str
    result: str
    similarity: float


def compute_similarity(query_embedding: np.ndarray, result_embedding: np.ndarray) -> float:
    """计算余弦相似度"""
    dot = np.dot(query_embedding, result_embedding)
    norm_q = np.linalg.norm(query_embedding)
    norm_r = np.linalg.norm(result_embedding)
    return dot / (norm_q * norm_r)


def select_top_k(query_embedding: np.ndarray, results: List[ScoredResult], top_k: int = 5) -> List[ScoredResult]:
    """筛选top-k最相关结果"""
    for result in results:
        result.similarity = compute_similarity(query_embedding, result.embedding)
    
    # 降序排序
    results_sorted = sorted(results, key=lambda x: x.similarity, reverse=True)
    
    # 返回top-k
    return results_sorted[:top_k]
