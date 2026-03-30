"""
attnres-multiagent - 注意力残差多智能体框架
===========================================

## 架构版本

### 1. v0: 原始attnres - Block注意力残差
- 基础Block分块聚合
- 解决上下文爆炸问题

### 2. 🔥 v1: HGARN - 层次化门控注意力残差网络 (创新!)
**全新创新架构**:
- 完整层次化残差连接，每个层次直接连接到最终输出
- 动态门控机制，自动过滤低信息增益结果
- **双向注意力流** —— 上层反向激活下层关键信息 (完全创新!)
- 置信度路由，自适应提前停止

Author: OpenClaw
License: MIT
"""

from attnres_multiagent import AttnResMultiAgent, RunResult
from hierarchical_attn_res import HGARMultiAgent
from attn_types import (
    SubTask,
    SubTaskResult,
    BlockResult,
    BlockAggregatedResult,
    DecompositionResult,
    HierarchicalLevel
)
from gated_residual_aggregator import GatedResidualAggregator
from vector_selector import compute_similarity, select_top_k, ScoredResult

__all__ = [
    # v0 原始架构 (兼容)
    'AttnResMultiAgent',
    # v1 创新架构 - 层次化门控注意力残差网络
    'HGARMultiAgent',
    # 类型定义
    'RunResult',
    'SubTask',
    'SubTaskResult',
    'BlockResult',
    'BlockAggregatedResult',
    'DecompositionResult',
    'HierarchicalLevel',
    'GatedResidualAggregator',
    # 工具函数
    'compute_similarity',
    'select_top_k',
    'ScoredResult',
]

__version__ = '1.0.0'

