"""
🔥 侧抑制 - 参数化自适应 (v2 新增)
===

改进设计:
1. 抑制强度自适应 - 结果越多，抑制越强
2. 相似度阈值可配置
3. 自动保持稀疏性，抑制高度相似结果

原理:
高度相似的结果会互相抑制，只有最相关的结果保留高分，
这样可以自动过滤冗余，保持表示的稀疏性。
"""

import math
import numpy as np
from typing import Optional


class AdaptiveLateralInhibition:
    """自适应侧抑制
    
    特性:
    - 抑制强度随结果数量增加而增加，自动保持稀疏
    - 相似度超过阈值才应用抑制
    - 可配置基础强度和阈值
    """
    
    def __init__(
        self,
        base_inhibition_strength: float = 0.1,
        similarity_threshold: float = 0.8,
        adaptive_strength: bool = True,
        min_inhibition: float = 0.0,
        max_inhibition: float = 0.5,
    ):
        """
        Args:
            base_inhibition_strength: 基础抑制强度，默认 0.1
            similarity_threshold: 相似度阈值，超过此值才抑制，默认 0.8
            adaptive_strength: 是否启用自适应强度，结果越多抑制越强
            min_inhibition: 最小抑制强度
            max_inhibition: 最大抑制强度，防止抑制过强
        """
        self.base_strength = base_inhibition_strength
        self.threshold = similarity_threshold
        self.adaptive = adaptive_strength
        self.min_inhibition = min_inhibition
        self.max_inhibition = max_inhibition
    
    def compute_inhibition_matrix(
        self,
        similarity_matrix: np.ndarray,
        num_results: int,
    ) -> np.ndarray:
        """计算侧抑制矩阵
        
        Args:
            similarity_matrix: 结果之间的相似度矩阵 (n x n)
            num_results: 结果数量
            
        Returns:
            inhibition_matrix: 抑制矩阵，每个元素表示要扣除的分数比例
            
        计算公式:
            如果 similarity > threshold:
                actual_strength = base_strength × (1 + ln(num_results))  (自适应开启)
            else:
                不抑制
        """
        n = num_results
        inhibition = np.zeros_like(similarity_matrix)
        
        # 计算实际抑制强度
        if self.adaptive and n > 1:
            # 结果越多，抑制越强，自动保持稀疏性
            actual_strength = self.base_strength * (1 + math.log(n))
        else:
            actual_strength = self.base_strength
        
        # 裁剪到合理范围
        actual_strength = max(self.min_inhibition, min(self.max_inhibition, actual_strength))
        
        # 应用抑制：相似度超过阈值的位置添加抑制
        mask = similarity_matrix > self.threshold
        inhibition[mask] = actual_strength
        
        # 对角线不抑制自己
        np.fill_diagonal(inhibition, 0.0)
        
        return inhibition
    
    def apply_inhibition(
        self,
        scores: np.ndarray,
        similarity_matrix: np.ndarray,
    ) -> np.ndarray:
        """应用侧抑制到原始分数
        
        Args:
            scores: 原始分数向量 (n,)
            similarity_matrix: 相似度矩阵 (n x n)
            
        Returns:
            inhibited_scores: 抑制后的分数
            
        公式:
            inhibited_scores[i] = scores[i] × (1 - sum_{j≠i} inhibition[i,j])
        """
        n = len(scores)
        inhibition = self.compute_inhibition_matrix(similarity_matrix, n)
        
        # 计算每个结果受到的总抑制
        total_inhibition = np.sum(inhibition, axis=1)
        
        # 应用抑制
        inhibited_scores = scores * (1.0 - total_inhibition)
        
        # 分数不能为负
        inhibited_scores = np.clip(inhibited_scores, 0.0, None)
        
        return inhibited_scores
    
    def compute_final_scores(
        self,
        raw_scores: np.ndarray,
        embeddings: np.ndarray,
        normalize: bool = True,
        enable_winner_take_all: bool = True,
        wta_top_k: int = 3,
    ) -> np.ndarray:
        """计算最终分数，完整流程
        
        v3 改进: 增加赢者通吃 (Winner-Take-All) 后处理
        
        Args:
            raw_scores: 原始分数 (n,)
            embeddings: 结果嵌入 (n x d)，用于计算相似度
            normalize: 是否归一化输出
            enable_winner_take_all: 是否启用赢者通吃，每个语义簇只保留Top-K
            wta_top_k: 每个簇保留多少个结果
            
        Returns:
            final_scores: 抑制后的最终分数
        """
        # 计算余弦相似度矩阵
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_embeddings = embeddings / (norm + 1e-8)
        similarity_matrix = norm_embeddings @ norm_embeddings.T
        
        # 应用侧抑制
        final_scores = self.apply_inhibition(raw_scores, similarity_matrix)
        
        # v3 改进: 赢者通吃后处理
        if enable_winner_take_all and len(final_scores) > wta_top_k:
            final_scores = self._apply_winner_take_all(
                final_scores, 
                similarity_matrix, 
                top_k=wta_top_k
            )
        
        # 可选归一化
        if normalize and np.sum(final_scores) > 0:
            final_scores = final_scores / np.sum(final_scores)
        
        return final_scores
    
    def _apply_winner_take_all(
        self,
        scores: np.ndarray,
        similarity_matrix: np.ndarray,
        top_k: int = 3,
    ) -> np.ndarray:
        """赢者通吃后处理 —— 每个相似度簇只保留Top-K得分最高的结果
        
        高度相似的结果中，只有得分最高的保留，其他被抑制到0
        这进一步增强稀疏性，去除冗余
        
        Args:
            scores: 侧抑制后的分数
            similarity_matrix: 相似度矩阵
            top_k: 每个连通分量保留多少个结果
            
        Returns:
            processed_scores: WTA处理后的分数
        """
        n = len(scores)
        result = scores.copy()
        
        # 连通分量聚类：相似度 > 阈值 => 同一个簇
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]:
                continue
            
            # BFS找连通分量
            from collections import deque
            queue = deque()
            queue.append(i)
            visited[i] = True
            cluster = []
            
            while queue:
                node = queue.popleft()
                cluster.append(node)
                
                # 找邻居：相似度 > 阈值且未访问
                for neighbor in range(n):
                    if not visited[neighbor] and similarity_matrix[node, neighbor] > self.threshold:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # 如果簇大小 > top_k，只保留Top-K得分最高的
            if len(cluster) > top_k:
                # 按分数排序
                cluster_scores = [(scores[idx], idx) for idx in cluster]
                cluster_scores.sort(reverse=True)
                
                # 抑制除了Top-K以外的所有
                for _, idx in cluster_scores[top_k:]:
                    result[idx] = 0.0
        
        return result


# 方便使用的默认实例
default_inhibitor = AdaptiveLateralInhibition()
