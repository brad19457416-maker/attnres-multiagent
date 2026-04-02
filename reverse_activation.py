"""
🔥 反向激活管理 (v2 新增)
===

改进设计:
1. 信息增益计算 - 预估增益，只有增益>阈值才触发
2. 批量处理待激活请求
3. 集成到工作记忆，增益过滤

双向注意力流核心:
- 传统: 下层 → 上层 单向流
- 我们: 上层发现新线索 → 反向激活下层相关信息
- v2 改进: 只在真正有信息增益时才激活，节省token
"""

from typing import List, Callable, Optional
from dataclasses import dataclass

from attn_types import (
    WorkingMemory,
    ReverseActivationRequest,
    HierarchicalLevel,
)


class ReverseActivationManager:
    """反向激活管理器
    
    职责:
    1. 收集反向激活请求
    2. 按增益阈值过滤
    3. 调用聚合器重新激活下层信息
    4. 更新工作记忆
    """
    
    def __init__(
        self,
        default_gain_threshold: float = 0.3,
    ):
        """
        Args:
            default_gain_threshold: 默认增益阈值，低于此值不触发
        """
        self.default_threshold = default_gain_threshold
    
    def get_pending_requests(
        self,
        working_memory: WorkingMemory,
        threshold: Optional[float] = None,
    ) -> List[ReverseActivationRequest]:
        """获取待处理的反向激活请求（过滤增益不足）"""
        if threshold is None:
            threshold = self.default_threshold
        return working_memory.get_pending_reverse_activations(threshold)
    
    def has_pending_requests(
        self,
        working_memory: WorkingMemory,
        threshold: Optional[float] = None,
    ) -> bool:
        """是否有待处理的反向激活请求"""
        pending = self.get_pending_requests(working_memory, threshold)
        return len(pending) > 0
    
    def reverse_activate(
        self,
        requests: List[ReverseActivationRequest],
        query: str,
        previous_levels: List[HierarchicalLevel],
        aggregate_fn: Callable[[List[ReverseActivationRequest], str, List[HierarchicalLevel]], str],
    ) -> str:
        """执行反向激活
        
        Args:
            requests: 待处理请求
            query: 原始查询
            previous_levels: 之前层次结果
            aggregate_fn: 重新聚合函数
            
        Returns:
            重新聚合后的结果
        """
        return aggregate_fn(requests, query, previous_levels)
    
    def clear_pending(
        self,
        working_memory: WorkingMemory,
    ):
        """清空已处理的请求"""
        working_memory.gating_area.clear_pending()


# 默认实例
default_manager = ReverseActivationManager()
