"""
类型定义
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SubTask:
    """子任务定义"""
    task_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    can_parallel: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0  # 递归深度，0表示顶层
    parent_task_id: Optional[str] = None  # 父任务ID


@dataclass
class DecompositionResult:
    """任务分解结果（包含递归分解）"""
    original_task: SubTask
    decomposed: bool  # 是否成功分解
    subtasks: List[SubTask] = field(default_factory=list)


@dataclass
class SubTaskResult:
    """子任务执行结果"""
    task_id: str
    task: SubTask
    result: str
    success: bool = True
    token_usage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockResult:
    """Block执行结果（聚合前）"""
    block_id: int
    subtasks: List[SubTask]
    results: List[SubTaskResult]
    start_idx: int
    end_idx: int


@dataclass
class BlockAggregatedResult:
    """Block聚合结果（聚合后）"""
    block_id: int
    aggregated_result: str
    attention_scores: Dict[str, float]  # task_id -> score
    total_token_usage: int
    original_count: int  # 原始子任务数量
    compressed: bool = True  # 是否压缩了


@dataclass
class BlockAggregatedResult:
    """Block聚合结果（聚合后）"""
    block_id: int
    aggregated_result: str
    attention_scores: Dict[str, float]  # task_id -> score
    total_token_usage: int
    original_count: int  # 原始子任务数量
    compressed: bool = True  # 是否压缩了
    gate_value: float = 1.0  # 门控值，用于层次残差连接
    residual_connection: bool = True  # 是否启用残差连接


@dataclass
class HierarchicalLevel:
    """层次化结构中的一层"""
    level_id: int
    blocks: List[BlockAggregatedResult]
    aggregated: str
    gate_score: float  # 该层整体门控分数


@dataclass
class RunResult:
    """完整运行结果"""
    query: str
    final_answer: str
    blocks_processed: int
    subtasks_total: int
    total_tokens: int
    blocks: List[BlockAggregatedResult] = field(default_factory=list)
    hierarchical_levels: List[HierarchicalLevel] = field(default_factory=list)
    success: bool = True
    early_stopped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
