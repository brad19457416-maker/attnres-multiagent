"""
attnres-multiagent - 注意力残差多智能体框架
===========================================

核心创新：将Kimi注意力残差(Attention Residuals)思想从神经网络层间
迁移到MultiAgent智能体层级间，通过Block注意力聚合解决
上下文爆炸和信息稀释问题。

Author: OpenClaw
Date: 2026-03-28
"""

from typing import List, Optional
from dataclasses import dataclass, field

from .attn_types import (
    SubTask, 
    SubTaskResult, 
    BlockResult, 
    BlockAggregatedResult,
    RunResult
)
from .task_decomposer import TaskDecomposer
from .subagent_executor import SubAgentExecutor
from .attention_aggregator import AttentionAggregator


class AttnResMultiAgent:
    """注意力残差多智能体
    
    核心架构：
    1. 任务分解 → 分解为多个子任务
    2. **递归分层** → 如果子任务仍然复杂，继续分解（可选）
    3. 分块 → 每个Block最多block_size个子任务
    4. 逐个Block处理:
       a. 并行/顺序执行子任务
       b. 注意力残差聚合 → 压缩token，保留重要信息
    5. 最终整合所有Block结果 → 输出回答
    """
    
    def __init__(self, 
                 block_size: int = 8,
                 max_blocks: int = 3,
                 adaptive_early_stop: bool = True,
                 parallel_execution: bool = False,
                 attn_score_model: str = "same",
                 enable_recursive_decomposition: bool = False,
                 max_recursion_depth: int = 3):
        """
        Args:
            block_size: 每个Block最多容纳多少个子任务
            max_blocks: 最多允许多少个Block（控制最大深度）
            adaptive_early_stop: 是否提前停止（MVP暂不支持）
            parallel_execution: 是否并行执行子任务（MVP暂不支持，顺序执行）
            attn_score_model: 用哪个模型计算注意力分数
            enable_recursive_decomposition: 是否启用递归分层分解
            max_recursion_depth: 最大递归深度（默认3层）
        """
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.adaptive_early_stop = adaptive_early_stop
        self.parallel_execution = parallel_execution
        self.attn_score_model = attn_score_model
        self.enable_recursive_decomposition = enable_recursive_decomposition
        self.max_recursion_depth = max_recursion_depth
        
        # 初始化模块
        self.decomposer = TaskDecomposer(max_recursion_depth=max_recursion_depth)
        self.executor = SubAgentExecutor(max_parallel=4 if parallel_execution else 1)
        self.aggregator = AttentionAggregator()
    
    def _flatten_recursive(self, query: str, tasks: List[SubTask]) -> List[SubTask]:
        """递归展开任务，直到不需要分解
        
        使用DFS递归分解，得到所有叶子节点任务。
        """
        result = []
        stack = list(tasks)
        
        while stack:
            task = stack.pop()
            
            if not self.enable_recursive_decomposition:
                result.append(task)
                continue
            
            # 检查是否需要进一步分解
            decomp_result = self.decomposer.check_need_decompose(task, query)
            
            if decomp_result.decomposed and decomp_result.subtasks:
                # 需要分解，将子任务推入栈继续处理
                for child in reversed(decomp_result.subtasks):
                    stack.append(child)
            else:
                # 不需要分解，加入结果
                result.append(task)
        
        return result
    
    def run(self, query: str) -> RunResult:
        """执行完整流程"""
        
        # 1. 顶层任务分解
        subtasks = self.decomposer.decompose(query)
        
        # 2. 递归分层分解（如果启用）
        if self.enable_recursive_decomposition:
            subtasks = self._flatten_recursive(query, subtasks)
        
        # 3. 分组为Block
        blocks = self.decomposer.group_into_blocks(subtasks, self.block_size)
        
        # 限制最大Block数量
        if len(blocks) > self.max_blocks:
            blocks = blocks[:self.max_blocks]
        
        # 4. 逐个Block处理
        processed_blocks: List[BlockAggregatedResult] = []
        total_tokens = 0
        previous_aggregated = ""
        
        for block_idx, block in enumerate(blocks):
            # a. 执行Block内所有子任务（如果启用并行执行且有多个任务则并行）
            use_parallel = self.parallel_execution and len(block) > 1
            block_result = self.executor.execute_block(
                block, 
                query,
                previous_aggregated,
                parallel=use_parallel,
                max_parallel=4
            )
            
            # b. 注意力残差聚合
            aggregated = self.aggregator.aggregate(
                block_result,
                query,
                processed_blocks
            )
            
            processed_blocks.append(aggregated)
            total_tokens += aggregated.total_token_usage
            
            # 更新previous_aggregated供下一Block参考
            previous_aggregated = "\n\n".join([
                b.aggregated_result for b in processed_blocks
            ])
            
            # 检查自适应提前停止
            if self.adaptive_early_stop and self._check_converged(processed_blocks, query):
                break
        
        # 5. 最终聚合
        final_answer = self.aggregator.final_aggregate(
            processed_blocks,
            query
        )
        
        return RunResult(
            query=query,
            final_answer=final_answer,
            blocks_processed=len(processed_blocks),
            subtasks_total=len(subtasks),
            total_tokens=total_tokens,
            blocks=processed_blocks,
            success=True,
            early_stopped=len(processed_blocks) < len(blocks),
            metadata={
                "recursive_decomposition_enabled": self.enable_recursive_decomposition,
                "max_recursion_depth": self.max_recursion_depth if self.enable_recursive_decomposition else 0
            }
        )
    
    def _check_converged(self, processed_blocks: List[BlockAggregatedResult], query: str) -> bool:
        """检查是否已经收敛，可以提前停止
        
        只有当adaptive_early_stop=True时才调用。
        """
        if not self.adaptive_early_stop or len(processed_blocks) == 0:
            return False
        
        # 如果已经达到max_blocks，不提前停止
        if len(processed_blocks) >= self.max_blocks:
            return False
        
        # 检查最后一个Block的注意力分数分布
        # 如果所有分数都很低，说明没有太多相关信息，可以提前停止
        last_block = processed_blocks[-1]
        scores = list(last_block.attention_scores.values())
        
        if not scores:
            return True
        
        avg_score = sum(scores) / len(scores)
        
        # 如果平均分低于3分（满分10），说明信息增益很小，可以停止
        if avg_score < 3.0:
            return True
        
        return False
