"""
🔥 Hierarchical Gated Attention Residual MultiAgent - 层次化门控注意力残差多智能体
===

**核心创新** (相对于原始attnres-multiagent):

1. **完整贯彻残差思想** —— 不仅仅Block内聚合，而是建立**层次化残差网络**
   - 每个Block → 残差连接到层次
   - 每个层次 → 残差连接到最终输出
   - 真正实现"每层都保留对最终输出的直接连接"，解决深层信息稀释

2. **动态门控机制** —— 每个Block/层次学习一个门控值
   - 门控值 = 信息增益置信度
   - 最终聚合按门控加权，自动抑制低质量结果
   - 自适应控制信息流

3. **双向注意力流** 🔥 完全创新!
   - 传统: 下层 → 上层 单向流
   - 我们: 支持上层结果**反向激活**下层关键信息
   - 上层发现新线索 → 唤醒下层相关信息重新强调
   - 类似于神经网络中注意力的反向传播加权

4. **置信度路由** —— 动态路径选择
   - 低置信度结果自动触发重新分解执行
   - 高置信度结果直接通过
   - 实现自我校正

论文创新对比:

| 方法 | 残差连接 | 门控机制 | 双向流 | Token复杂度 |
|------|----------|----------|--------|-------------|
| 原始MoA | ❌ | ❌ | ❌ | O(N) |
| Attention-MoA | ❌ | ❌ | ❌ | O(N) |
| Kimi Attention Residual | ✅ (神经网络层) | ❌ | ❌ | O(L*D) |
| attnres-multiagent v0 | ✅ (Block间拼接) | ❌ | ❌ | O(B) |
| **HGARN (我们)** | ✅ (完整层次残差网络) | ✅ 动态门控 | ✅ 双向注意力流 | **O(L) ≈ 常数** |

> **L = 层次数，B = Block数，N = 子任务数，L ≤ B << N**

设计: 燕鱼 & 火山大龙虾
创新: 双向注意力流 + 层次化门控残差网络
"""

from typing import List, Optional, Callable

from attn_types import (
    SubTask, 
    DecompositionResult,
    SubTaskResult, 
    BlockResult, 
    BlockAggregatedResult,
    HierarchicalLevel,
    RunResult,
    WorkingMemory,
    ReverseActivationRequest,
)
from task_decomposer import TaskDecomposer
from subagent_executor import SubAgentExecutor
from gated_residual_aggregator import GatedResidualAggregator
from reverse_activation import ReverseActivationManager
from concurrency_controller import DynamicConcurrencyController, RetryPolicy


class HGARMultiAgent:
    """🔥 Hierarchical Gated Attention Residual MultiAgent
    层次化门控注意力残差多智能体 —— 完整创新架构
    
    核心架构:
    ```
    用户查询
        │
        ▼
    任务分解 → 递归分解 → 分组Block → 分组层次
        │
        ▼
    逐层次处理 (每个层次包含多个Block):
        每个Block:
        → 执行子任务
        → 门控注意力聚合 (打分 + 门控值 + 反向激活主题)
        ↓
        层次内聚合Block
        → 如果有反向激活需求 → 反向激活下层 → 更新下层信息
        ↓
        层次结果通过残差连接到最终输出
        │
        ▼
    最终聚合: 按门控加权整合所有层次残差 → 输出回答
    ```
    """
    
    def __init__(self, 
                 block_size: int = 8,
                 max_blocks_per_level: int = 2,
                 max_levels: int = 3,
                 enable_recursive_decomposition: bool = True,
                 max_recursion_depth: int = 3,
                 parallel_execution: bool = True,
                 max_parallel: int = 4,
                 enable_reverse_activation: bool = True,
                 enable_confidence_routing: bool = True,
                 min_gate_for_continue: float = 0.15,
                 # v2 新增参数
                 enable_working_memory_partition: bool = True,
                 gate_at_block_level: bool = False,
                 reverse_activation_gain_threshold: float = 0.3,
                 enable_dynamic_concurrency: bool = True,
                 min_concurrency: int = 1,
                 max_concurrency: int = 8,
                 # v3 优化新增: 累积增益早停
                 cumulative_gain_threshold: float = 2.5,
                 # v3 优化新增: 向量预过滤
                 enable_vector_prefilter: bool = True,
                 vector_similarity_threshold: float = 0.3,
                 llm_client: Callable = None):
        """
        Args:
            block_size: 每个Block最多子任务数 (默认 8)
            max_blocks_per_level: 每个层次最多Block数 (默认 2)
            max_levels: 最大层次数 (默认 3)，token ≈ 常数 × max_levels
            enable_recursive_decomposition: 是否启用递归分解
            max_recursion_depth: 最大递归深度
            parallel_execution: 是否启用并行执行
            max_parallel: 最大并行数（固定并发时）
            enable_reverse_activation: 是否启用双向注意力流反向激活
            enable_confidence_routing: 是否启用置信度路由（低置信度重跑）
            min_gate_for_continue: 最小门控值继续下一层，如果最后一层平均门控低于此值提前停止
            
            # v2 新增参数
            enable_working_memory_partition: 是否启用工作记忆分区 (v2 改进)
            gate_at_block_level: 是否在Block级别计算门控，False 只在层次级别计算 → 节省token
            reverse_activation_gain_threshold: 反向激活增益阈值，低于此不触发
            enable_dynamic_concurrency: 是否启用动态并发控制 (v2 改进)
            min_concurrency: 动态并发最小并发数
            max_concurrency: 动态并发最大并发数
            
            llm_client: LLM调用函数，如果不提供使用全局call_llm
        """
        self.block_size = block_size
        self.max_blocks_per_level = max_blocks_per_level
        self.max_levels = max_levels
        self.enable_recursive_decomposition = enable_recursive_decomposition
        self.max_recursion_depth = max_recursion_depth
        self.parallel_execution = parallel_execution
        self.max_parallel = max_parallel
        self.enable_reverse_activation = enable_reverse_activation
        self.enable_confidence_routing = enable_confidence_routing
        self.min_gate_for_continue = min_gate_for_continue
        
        # v2 配置
        self.enable_working_memory = enable_working_memory_partition
        self.reverse_gain_threshold = reverse_activation_gain_threshold
        # v3 优化: 累积增益早停
        self.cumulative_gain_threshold = cumulative_gain_threshold
        # v3 优化: 向量预过滤
        self.enable_vector_prefilter = enable_vector_prefilter
        self.vector_similarity_threshold = vector_similarity_threshold
        
        # 初始化模块
        self.decomposer = TaskDecomposer(
            llm_client=llm_client, 
            max_recursion_depth=max_recursion_depth
        )
        # v2: 动态并发控制
        if enable_dynamic_concurrency:
            max_retries = 3
        else:
            max_retries = 2 if enable_confidence_routing else 1
            
        self.executor = SubAgentExecutor(
            max_parallel=max_parallel if parallel_execution else 1,
            max_retries=max_retries,
            llm_client=llm_client,
            enable_dynamic_concurrency=enable_dynamic_concurrency,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
        )
        self.aggregator = GatedResidualAggregator(
            llm_client=llm_client,
            gate_at_block_level=gate_at_block_level,
            reverse_activation_gain_threshold=reverse_activation_gain_threshold,
            enable_vector_prefilter=enable_vector_prefilter if 'enable_vector_prefilter' in locals() else True,
            vector_similarity_threshold=vector_similarity_threshold if 'vector_similarity_threshold' in locals() else 0.3,
        )
        # v2 新增模块
        self.reverse_manager = ReverseActivationManager(
            default_gain_threshold=reverse_activation_gain_threshold
        )
        if enable_dynamic_concurrency:
            self.concurrency_controller = DynamicConcurrencyController(
                min_concurrency=min_concurrency,
                max_concurrency=max_concurrency,
                priority_by_gate=True
            )
    
    def _flatten_recursive(self, query: str, tasks: List[SubTask]) -> List[SubTask]:
        """递归展开任务"""
        return self.decomposer.flatten_recursive_tasks(tasks, query)
    
    def _group_into_blocks(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        """将子任务分组为Block"""
        return self.decomposer.group_into_blocks(subtasks, self.block_size)
    
    def _group_blocks_into_levels(self, blocks: List[List[SubTask]]) -> List[List[List[SubTask]]]:
        """将Block分组为层次"""
        levels = []
        current_level_blocks = []
        
        for block in blocks:
            if len(current_level_blocks) >= self.max_blocks_per_level:
                levels.append(current_level_blocks)
                current_level_blocks = []
            current_level_blocks.append(block)
        
        if current_level_blocks:
            levels.append(current_level_blocks)
        
        # 限制最大层次数
        if len(levels) > self.max_levels:
            levels = levels[:self.max_levels]
        
        return levels
    
    def run(self, query: str) -> RunResult:
        """执行完整的层次化门控注意力残差多智能体流程 (v2 更新)
        
        v2 改进:
        - 集成工作记忆分区
        - 支持反向激活增益过滤
        - 支持动态并发控制
        """
        
        # ========== Step 1: 初始化工作记忆 (v2 新增) ==========
        if self.enable_working_memory:
            wm = WorkingMemory.create(query)
        else:
            wm = None
        
        # ========== Step 2: 任务分解 ==========
        # 顶层分解 + 递归展开
        subtasks = self.decomposer.decompose(query)
        
        if self.enable_recursive_decomposition:
            subtasks = self._flatten_recursive(query, subtasks)
        
        total_subtasks = len(subtasks)
        
        # ========== Step 3: 分组 ==========
        # 子任务 → Block → 层次
        blocks = self._group_into_blocks(subtasks)
        levels_blocked = self._group_blocks_into_levels(blocks)
        
        # ========== Step 4: 逐层次处理 ==========
        processed_levels: List[HierarchicalLevel] = []
        total_tokens = 0
        block_counter = 0
        cumulative_gain = 0.0
        
        for level_idx, level_blocks in enumerate(levels_blocked):
            # 处理当前层次中的每个Block
            processed_blocks: List[BlockAggregatedResult] = []
            
            for block_idx, block_subtasks in enumerate(level_blocks):
                # a. 执行Block内所有子任务
                block_id = block_counter
                block_counter += 1
                
                # 获取上下文从工作记忆
                if wm:
                    prev_aggregated = wm.get_full_context()
                else:
                    prev_aggregated = "\n\n".join([
                        f"Level {lv.level_id}:\n{lv.aggregated}" 
                        for lv in processed_levels
                    ])
                
                # 判断是否并行（v2: 动态并发）
                if hasattr(self, 'concurrency_controller'):
                    # 使用动态并发控制器
                    self.concurrency_controller.reset_stats()
                    # 这里简化，并发数由控制器动态计算
                    current_concurrency = self.concurrency_controller.get_current_concurrency([])
                    use_parallel = self.parallel_execution and len(block_subtasks) > 1
                    actual_parallel = min(current_concurrency, len(block_subtasks)) if use_parallel else 1
                else:
                    use_parallel = self.parallel_execution and len(block_subtasks) > 1
                    actual_parallel = self.max_parallel if use_parallel else 1
                
                block_result = self.executor.execute_block(
                    block_subtasks,
                    query,
                    previous_blocks_aggregated=prev_aggregated,
                    parallel=use_parallel,
                    max_parallel=actual_parallel
                )
                # 覆盖block_id
                block_result.block_id = block_id
                
                # b. 门控注意力聚合
                aggregated_block = self.aggregator.aggregate_block(
                    block_result,
                    query,
                    processed_levels
                )
                
                # v2: 如果启用工作记忆，添加反向激活请求
                if wm and self.enable_reverse_activation:
                    # 从聚合结果提取反向激活请求（这里需要从LLM输出获取）
                    # 实际已经在aggregate_block中解析，现在添加到工作记忆
                    # 注意：当前实现中，reverse_activation_topics 需要从data获取
                    # 这里我们保持兼容，后续完整实现
                    pass
                
                processed_blocks.append(aggregated_block)
                total_tokens += aggregated_block.total_token_usage
            
            # 层次内聚合多个Block
            current_level = self.aggregator.aggregate_level(
                processed_blocks,
                level_id=level_idx,
                query=query,
                previous_levels=processed_levels
            )
            
            # 更新累积增益
            cumulative_gain += current_level.gate_score
            
            # 🔥 v2 改进: 双向注意力流 - 反向激活下层，带增益过滤
            if wm and self.enable_reverse_activation and len(processed_levels) > 0:
                if self.reverse_manager.has_pending_requests(wm, self.reverse_gain_threshold):
                    pending = self.reverse_manager.get_pending_requests(wm, self.reverse_gain_threshold)
                    # 执行反向激活
                    if len(pending) > 0:
                        revived_result = self.aggregator.reverse_activate_lower(
                            [t.__dict__ for t in pending],
                            query,
                            processed_levels
                        )
                        # 将反向激活结果合并到当前层次
                        current_level.aggregated = current_level.aggregated + "\n\n**反向激活补充:**\n" + revived_result
                    self.reverse_manager.clear_pending(wm)
            
            # 添加到处理完的层次和工作记忆
            processed_levels.append(current_level)
            if wm:
                wm.commit_level_result(current_level)
            
            # 🔥 创新: 置信度路由 - 基于门控置信度提前停止
            # v3 改进: 累积增益判断，达到阈值就停止
            if self.enable_confidence_routing and level_idx < self.max_levels - 1:
                if current_level.gate_score < self.min_gate_for_continue:
                    # 信息增益太小，提前停止
                    break
                if cumulative_gain >= self.cumulative_gain_threshold:
                    # 已经获得足够累积增益，提前停止
                    break
        
        # ========== Step 5: 最终聚合 ==========
        # 整合所有层次的残差连接，按门控加权
        final_answer = self.aggregator.final_aggregate(
            processed_levels,
            query
        )
        
        # 收集所有Block
        all_blocks = []
        for lv in processed_levels:
            all_blocks.extend(lv.blocks)
        
        # 构建元数据（包含v2特性）
        metadata = {
            "architecture": "HGARN v2 - Hierarchical Gated Attention Residual Network",
            "max_levels": self.max_levels,
            "block_size": self.block_size,
            "enable_reverse_activation": self.enable_reverse_activation,
            "enable_confidence_routing": self.enable_confidence_routing,
            "enable_recursive_decomposition": self.enable_recursive_decomposition,
            "enable_working_memory_partition": self.enable_working_memory,
            "reverse_activation_gain_threshold": self.reverse_gain_threshold,
        }
        
        if hasattr(self, 'concurrency_controller'):
            metadata["enable_dynamic_concurrency"] = True
        
        return RunResult(
            query=query,
            final_answer=final_answer,
            blocks_processed=block_counter,
            subtasks_total=total_subtasks,
            total_tokens=total_tokens,
            blocks=all_blocks,
            hierarchical_levels=processed_levels,
            success=True,
            early_stopped=len(processed_levels) < len(levels_blocked),
            metadata=metadata
        )
