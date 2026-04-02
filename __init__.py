"""
attnres-multiagent - 注意力残差多智能体框架
===========================================

## 架构版本

### 1. v0: 原始attnres - Block注意力残差
- 基础Block分块聚合
- 解决上下文爆炸问题

### 2. 🔥 v1/v2: HGARN - 层次化门控注意力残差网络 (创新!)
**全新创新架构**:
- 完整层次化残差连接，每个层次直接连接到最终输出
- 动态门控机制，自动过滤低信息增益结果
- **双向注意力流** —— 上层反向激活下层关键信息 (完全创新!)
- 置信度路由，自适应提前停止

v2 改进:
- ✅ 工作记忆分区设计，减少干扰
- ✅ 自适应侧抑制，自动稀疏化
- ✅ 动态并发控制 + 指数退避重试
- ✅ 反向激活增益过滤，节省token
- ✅ 支持外部 LLM 客户端注入，兼容千帆/OpenAI等

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
    HierarchicalLevel,
    WorkingMemory,
    ReverseActivationRequest,
    Skill,
    TaskWithPriority,
)
from gated_residual_aggregator import GatedResidualAggregator
from lateral_inhibition import AdaptiveLateralInhibition
from concurrency_controller import DynamicConcurrencyController, RetryPolicy
from skill_forgetting import AdaptiveSkillForgetting
from reverse_activation import ReverseActivationManager
from llm_client_base import LLMClient, QianfanCodingPlanClient
from vector_selector import compute_similarity, select_top_k, ScoredResult

__all__ = [
    # v0 原始架构 (兼容)
    'AttnResMultiAgent',
    # HGARN 创新架构 v2
    'HGARMultiAgent',
    # 类型定义
    'RunResult',
    'SubTask',
    'SubTaskResult',
    'BlockResult',
    'BlockAggregatedResult',
    'DecompositionResult',
    'HierarchicalLevel',
    # v2 新增类型
    'WorkingMemory',
    'ReverseActivationRequest',
    'Skill',
    'TaskWithPriority',
    # 模块
    'GatedResidualAggregator',
    'AdaptiveLateralInhibition',
    'DynamicConcurrencyController',
    'RetryPolicy',
    'AdaptiveSkillForgetting',
    'ReverseActivationManager',
    # LLM 客户端支持
    'LLMClient',
    'QianfanCodingPlanClient',
    # 工具函数
    'compute_similarity',
    'select_top_k',
    'ScoredResult',
]

__version__ = '2.0.0'


