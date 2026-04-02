"""
子任务执行器 - 执行子Agent任务 (v2 更新)

v2 改进:
- 支持动态并发控制器
- 支持指数退避重试
- 优先级调度
"""

from typing import List, Callable, Optional
import time

import concurrent.futures

from attn_types import SubTask, SubTaskResult, BlockResult, TaskWithPriority
from concurrency_controller import DynamicConcurrencyController, RetryPolicy


class SubAgentExecutor:
    """子任务执行器，支持顺序和并行执行，内置容错重试机制
    
    v2 改进:
    - 可选动态并发控制
    - 指数退避重试
    - 优先级调度（高门控优先）
    """
    
    def __init__(
        self,
        max_parallel: int = 4,
        max_retries: int = 3,
        llm_client: Callable = None,
        enable_dynamic_concurrency: bool = True,
        min_concurrency: int = 1,
        max_concurrency: int = 8,
    ):
        """
        Args:
            max_parallel: 最大并行数（固定并发模式）
            max_retries: 最大重试次数
            llm_client: LLM调用函数
            enable_dynamic_concurrency: 是否启用动态并发控制 (v2 新增)
            min_concurrency: 动态并发最小并发数
            max_concurrency: 动态并发最大并发数
        """
        self.max_parallel = max_parallel
        self.max_retries = max_retries  # 失败后最大重试次数
        self._call_llm = llm_client if llm_client else None
        self.enable_dynamic = enable_dynamic_concurrency
        
        if enable_dynamic_concurrency:
            self.dynamic_controller = DynamicConcurrencyController(
                min_concurrency=min_concurrency,
                max_concurrency=max_concurrency,
                priority_by_gate=True,
                retry_policy=RetryPolicy(
                    max_retries=max_retries,
                    initial_backoff_ms=1000,
                    backoff_multiplier=2.0,
                    max_backoff_ms=30000
                )
            )
        else:
            self.dynamic_controller = None
    
    def execute_subtask(self, subtask: SubTask, 
                       query: str, 
                       previous_results: str = "",
                       retry_policy: Optional[RetryPolicy] = None) -> SubTaskResult:
        """执行单个子任务，带容错重试
        
        v2 改进: 支持指数退避重试
        """
        
        # 构建prompt
        if previous_results:
            prompt = f"""原始用户查询: {query}

之前聚合结果参考:
{previous_results}

请你执行以下子任务:
{subtask.description}

给出你的回答:"""
        else:
            prompt = f"""原始用户查询: {query}

请你执行以下子任务:
{subtask.description}

给出你的回答:"""
        
        # 使用默认策略如果没提供
        if retry_policy is None:
            retry_policy = RetryPolicy(max_retries=self.max_retries)
        
        # 重试循环
        last_exception = None
        for retry_count in range(retry_policy.max_retries + 1):
            try:
                # 重试时温度稍高一点，增加跳出概率
                temperature = 0.7 if retry_count == 0 else 0.9
                if self._call_llm:
                    result = self._call_llm(prompt, temperature=temperature)
                else:
                    global call_llm
                    result = call_llm(prompt, temperature=temperature)
                
                if result and len(result.strip()) > 0:
                    # 执行成功
                    estimated_tokens = len(result) // 4
                    return SubTaskResult(
                        task_id=subtask.task_id,
                        task=subtask,
                        result=result.strip(),
                        success=True,
                        token_usage=estimated_tokens
                    )
            except Exception as e:
                last_exception = e
                # 指数退避等待
                if retry_policy.should_retry(retry_count):
                    retry_policy.wait(retry_count)
                # 继续重试
                continue
        
        # 所有重试都失败了
        error_msg = str(last_exception) if last_exception else "空结果"
        return SubTaskResult(
            task_id=subtask.task_id,
            task=subtask,
            result=f"子任务执行失败: {error_msg}",
            success=False,
            token_usage=0
        )
    
    def execute_block(self, block: List[SubTask], 
                     query: str,
                     previous_blocks_aggregated: str = "",
                     parallel: bool = False,
                     max_parallel: int = 4) -> BlockResult:
        """执行一个Block内的所有子任务
        
        Args:
            block: 要执行的子任务列表
            query: 原始用户查询
            previous_blocks_aggregated: 之前Block的聚合结果
            parallel: 是否并行执行
            max_parallel: 最大并行数 (固定并发模式)
        
        v2 改进:
        - 如果启用动态并发，使用动态并发控制器
        - 支持优先级调度（高门控优先）
        """
        expected_gates = [getattr(t, 'expected_gate', 0.5) for t in block]
        
        if self.enable_dynamic and self.dynamic_controller and parallel and len(block) > 1:
            # 使用动态并发控制器 (v2)
            prioritized = self.dynamic_controller.schedule_tasks(block, expected_gates)
            current_concurrency = self.dynamic_controller.get_current_concurrency(prioritized)
            actual_parallel = min(current_concurrency, len(prioritized))
            # 提取排序后的任务
            sorted_tasks = [p.task for p in prioritized]
            # 动态并发执行
            results = self._execute_block_parallel_dynamic(
                sorted_tasks,
                query,
                previous_blocks_aggregated,
                actual_parallel
            )
        elif parallel and len(block) > 1:
            # 固定并发（兼容v1）
            results = self._execute_block_parallel(
                block, 
                query, 
                previous_blocks_aggregated,
                max_parallel
            )
        else:
            # 顺序执行
            results = []
            retry_policy = None
            if self.enable_dynamic and self.dynamic_controller:
                retry_policy = self.dynamic_controller.retry_policy
            for subtask in block:
                result = self.execute_subtask(
                    subtask, 
                    query, 
                    previous_results=previous_blocks_aggregated,
                    retry_policy=retry_policy
                )
                if self.enable_dynamic and self.dynamic_controller:
                    self.dynamic_controller.on_task_complete(result.success)
                results.append(result)
        
        # 获取block id（简单处理，用id首字母）
        block_id = f"block_{block[0].task_id.split('_')[1]}" if '_' in block[0].task_id and len(block) > 0 else f"block_{hash(str(block)) % 1000}"
        
        return BlockResult(
            block_id=block_id,
            subtasks=block,
            results=results,
            start_idx=0,
            end_idx=len(block)-1
        )
    
    def _execute_block_parallel(self, block: List[SubTask],
                               query: str,
                               previous_results: str,
                               max_parallel: int) -> List[SubTaskResult]:
        """并行执行Block内可并行的任务 (固定并发)"""
        
        # 分离可并行和不可并行
        parallel_tasks = [t for t in block if t.can_parallel]
        serial_tasks = [t for t in block if not t.can_parallel]
        
        results = []
        
        # 先并行执行可并行任务
        if parallel_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_parallel, len(parallel_tasks))) as executor:
                # 提交所有任务
                futures = []
                for task in parallel_tasks:
                    future = executor.submit(
                        self.execute_subtask,
                        task,
                        query,
                        previous_results
                    )
                    futures.append(future)
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # 执行失败，标记为失败但继续
                        # 找到对应的task（这里顺序可能不对，但不影响，因为最后按task_id排序）
                        task = None
                        for t in parallel_tasks:
                            if not any(r.task_id == t.task_id for r in results):
                                task = t
                                break
                        if task:
                            results.append(SubTaskResult(
                                task_id=task.task_id,
                                task=task,
                                result=f"执行失败: {str(e)}",
                                success=False
                            ))
        
        # 顺序执行不可并行任务
        for task in serial_tasks:
            result = self.execute_subtask(task, query, previous_results)
            if self.enable_dynamic and self.dynamic_controller:
                self.dynamic_controller.on_task_complete(result.success)
            results.append(result)
        
        # 按原始任务顺序排序结果
        all_tasks = parallel_tasks + serial_tasks
        task_id_to_result = {r.task_id: r for r in results}
        ordered_results = [task_id_to_result[t.task_id] for t in all_tasks if t.task_id in task_id_to_result]
        
        return ordered_results
    
    def _execute_block_parallel_dynamic(
        self,
        block: List[SubTask],
        query: str,
        previous_results: str,
        max_parallel: int,
    ) -> List[SubTaskResult]:
        """动态并发执行Block内可并行的任务 (v2 新增)
        
        特点:
        - 优先级排序，高门控优先
        - 动态并发批处理
        - 指数退避重试
        - 失败率统计
        """
        # 分离可并行和不可并行
        parallel_tasks = [t for t in block if t.can_parallel]
        serial_tasks = [t for t in block if not t.can_parallel]
        
        results = []
        retry_policy = self.dynamic_controller.retry_policy if self.dynamic_controller else None
        
        # 先并行执行可并行任务（分批动态）
        if parallel_tasks:
            remaining = list(parallel_tasks)  # 复制一份，剩余任务
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                while remaining:
                    # 获取当前并发批大小
                    current_concurrency = self.dynamic_controller.get_current_concurrency([]) if self.dynamic_controller else max_parallel
                    batch = remaining[:current_concurrency]
                    remaining = remaining[current_concurrency:]
                    
                    # 提交当前批次
                    futures = {}
                    for task in batch:
                        future = executor.submit(
                            self.execute_subtask,
                            task,
                            query,
                            previous_results,
                            retry_policy
                        )
                        futures[future] = task
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(futures):
                        task = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            # 执行失败
                            result = SubTaskResult(
                                task_id=task.task_id,
                                task=task,
                                result=f"执行失败: {str(e)}",
                                success=False
                            )
                            results.append(result)
                        
                        # 更新统计
                        if self.dynamic_controller:
                            self.dynamic_controller.on_task_complete(result.success)
        
        # 顺序执行不可并行任务
        for task in serial_tasks:
            result = self.execute_subtask(task, query, previous_results, retry_policy)
            if self.dynamic_controller:
                self.dynamic_controller.on_task_complete(result.success)
            results.append(result)
        
        # 按原始任务顺序排序结果
        all_tasks = parallel_tasks + serial_tasks
        task_id_to_result = {r.task_id: r for r in results}
        ordered_results = [task_id_to_result[t.task_id] for t in all_tasks if t.task_id in task_id_to_result]
        
        return ordered_results
