"""
🔥 动态并发控制器 (v2 新增)
===

改进设计:
1. 动态调整并发数 - 根据失败率自动调整
2. 优先级调度 - 高门控任务优先执行
3. 指数退避重试 - 失败自动重试，退避时间指数增长
4. 高门控任务优先级更高，更早出结果

适用场景:
- API调用限流，需要动态调整并发
- 不同置信度任务需要区分优先级
- 失败后自动重试提高成功率
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any

from attn_types import SubTask, TaskWithPriority


@dataclass
class RetryPolicy:
    """失败重试策略 - 指数退避"""
    max_retries: int = 3
    initial_backoff_ms: int = 1000
    backoff_multiplier: float = 2.0
    max_backoff_ms: int = 30000
    
    def get_backoff_ms(self, retry_count: int) -> int:
        """计算本次重试的退避时间
        
        Args:
            retry_count: 已经重试了几次 (0 表示第一次失败)
            
        Returns:
            退避时间毫秒数
        """
        backoff = self.initial_backoff_ms * (self.backoff_multiplier ** retry_count)
        return int(min(backoff, self.max_backoff_ms))
    
    def should_retry(self, retry_count: int) -> bool:
        """是否应该继续重试"""
        return retry_count < self.max_retries
    
    def wait(self, retry_count: int):
        """等待退避时间"""
        backoff_ms = self.get_backoff_ms(retry_count)
        time.sleep(backoff_ms / 1000.0)


@dataclass
class TaskQueueStats:
    """任务队列统计"""
    total_pending: int = 0
    completed: int = 0
    failed: int = 0
    recent_fail_rate: float = 0.0
    
    def update_fail_rate(self, window_size: int = 10):
        """更新最近失败率"""
        # 实际实现可以滑动窗口，这里简化
        total = self.completed + self.failed
        if total > 0:
            self.recent_fail_rate = self.failed / total
        else:
            self.recent_fail_rate = 0.0


class DynamicConcurrencyController:
    """动态并发控制器
    
    特性:
    - 动态并发数：失败率越高，并发数越小，避免雪崩
    - 优先级调度：高预估门控任务优先执行
    - 支持重试策略：指数退避自动重试
    """
    
    def __init__(
        self,
        min_concurrency: int = 1,
        max_concurrency: int = 8,
        priority_by_gate: bool = True,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """
        Args:
            min_concurrency: 最小并发数
            max_concurrency: 最大并发数
            priority_by_gate: 是否按预估门控排序，高门控优先
            retry_policy: 重试策略，默认使用标准指数退避
        """
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.priority_by_gate = priority_by_gate
        self.retry_policy = retry_policy or RetryPolicy()
        self.stats = TaskQueueStats()
    
    def get_current_concurrency(
        self,
        pending_tasks: List[TaskWithPriority],
        recent_fail_rate: Optional[float] = None,
    ) -> int:
        """计算当前应该使用的并发数
        
        公式:
            实际并发 = min_concurrency + (max_concurrency - min_concurrency) × (1 - fail_rate)
            
        - 失败率越高 → 并发越小
        - 高门控任务多 → 适当提高并发
        
        Args:
            pending_tasks: 待处理任务列表
            recent_fail_rate: 最近失败率，如果 None 则使用统计值
            
        Returns:
            建议的并发数
        """
        if recent_fail_rate is None:
            recent_fail_rate = self.stats.recent_fail_rate
        
        # 基于失败率计算基础并发
        base_concurrency = self.min_concurrency + \
            (self.max_concurrency - self.min_concurrency) * (1 - recent_fail_rate)
        
        # 如果启用优先级调度，高门控任务多可以适当提高并发
        if self.priority_by_gate and len(pending_tasks) > 0:
            high_gate_count = sum(1 for t in pending_tasks if t.expected_gate > 0.5)
            # 每有一个高门控任务，增加 10% 并发
            adjustment = 1.0 + 0.1 * high_gate_count
            base_concurrency = base_concurrency * adjustment
        
        # 裁剪到范围内
        base_concurrency = int(math.ceil(base_concurrency))
        return max(self.min_concurrency, min(self.max_concurrency, base_concurrency))
    
    def schedule_tasks(self, tasks: List[SubTask], expected_gates: Optional[List[float]] = None) -> List[TaskWithPriority]:
        """任务调度 → 按优先级排序
        
        Args:
            tasks: 待处理子任务列表
            expected_gates: 每个任务的预估门控分数，如果没有则默认 0.5
            
        Returns:
            排序后的带优先级任务列表，高优先级在前
        """
        if expected_gates is None:
            expected_gates = [0.5] * len(tasks)
        
        # 包装为带优先级的任务
        prioritized = []
        for task, gate in zip(tasks, expected_gates):
            # 优先级 = 预估门控
            priority = gate
            prioritized.append(TaskWithPriority(
                task=task,
                priority=priority,
                expected_gate=gate
            ))
        
        # 按优先级降序排序 → 高优先级先执行
        if self.priority_by_gate:
            prioritized.sort(reverse=True, key=lambda x: x.priority)
        
        return prioritized
    
    def get_batch_to_run(self, pending: List[TaskWithPriority], recent_fail_rate: Optional[float] = None) -> List[TaskWithPriority]:
        """获取一批要运行的任务
        
        Args:
            pending: 待处理任务列表
            recent_fail_rate: 最近失败率
            
        Returns:
            这一轮要运行的任务批次
        """
        concurrency = self.get_current_concurrency(pending, recent_fail_rate)
        batch_size = min(concurrency, len(pending))
        return pending[:batch_size]
    
    def on_task_complete(self, success: bool):
        """任务完成回调，更新统计"""
        if success:
            self.stats.completed += 1
        else:
            self.stats.failed += 1
        self.stats.update_fail_rate()
    
    def reset_stats(self):
        """重置统计，开始新的运行"""
        self.stats = TaskQueueStats()


# 默认实例
default_controller = DynamicConcurrencyController()
