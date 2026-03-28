"""
attnres-multiagent - 注意力残差多智能体框架
===========================================

基于注意力残差(Attention Residuals)思想的MultiAgent架构创新，
通过Block注意力聚合解决上下文爆炸和信息稀释问题。

Author: OpenClaw
License: MIT
"""

from .attnres_multiagent import AttnResMultiAgent, RunResult
from .attn_types import (
    SubTask, 
    SubTaskResult, 
    BlockResult, 
    BlockAggregatedResult,
    DecompositionResult
)
from .vector_selector import compute_similarity, select_top_k, ScoredResult

__all__ = [
    'AttnResMultiAgent',
    'RunResult',
    'SubTask',
    'SubTaskResult',
    'BlockResult',
    'BlockAggregatedResult',
    'DecompositionResult',
    'compute_similarity',
    'select_top_k',
    'ScoredResult',
]

__version__ = '0.1.0'
