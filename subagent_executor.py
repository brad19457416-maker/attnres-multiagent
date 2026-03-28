"""
子任务执行器 - 执行子Agent任务
"""

from typing import List
from dataclasses import dataclass

from .attn_types import SubTask, SubTaskResult, BlockResult


import concurrent.futures
from typing import List, Tuple

from .attn_types import SubTask, SubTaskResult, BlockResult


class SubAgentExecutor:
    """子任务执行器，支持顺序和并行执行"""
    
    def __init__(self, max_parallel: int = 4):
        self.max_parallel = max_parallel
    
    def execute_subtask(self, subtask: SubTask, 
                       query: str, 
                       previous_results: str = "") -> SubTaskResult:
        """执行单个子任务"""
        
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
        
        # call_llm 由外部注入
        global call_llm
        result = call_llm(prompt, temperature=0.7)
        
        # 这里简化处理，token_usage估算
        estimated_tokens = len(result) // 4
        
        return SubTaskResult(
            task_id=subtask.task_id,
            task=subtask,
            result=result.strip(),
            success=True,
            token_usage=estimated_tokens
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
            max_parallel: 最大并行数
        """
        
        if parallel and len(block) > 1:
            # 并行执行可并行的任务
            results = self._execute_block_parallel(
                block, 
                query, 
                previous_blocks_aggregated,
                max_parallel
            )
        else:
            # 顺序执行
            results = []
            for subtask in block:
                result = self.execute_subtask(
                    subtask, 
                    query, 
                    previous_results=previous_blocks_aggregated
                )
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
        """并行执行Block内可并行的任务"""
        
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
                        task = parallel_tasks[len(results)]
                        results.append(SubTaskResult(
                            task_id=task.task_id,
                            task=task,
                            result=f"执行失败: {str(e)}",
                            success=False
                        ))
        
        # 顺序执行不可并行任务
        for task in serial_tasks:
            result = self.execute_subtask(task, query, previous_results)
            results.append(result)
        
        # 按原始任务顺序排序结果
        all_tasks = parallel_tasks + serial_tasks
        task_id_to_result = {r.task_id: r for r in results}
        ordered_results = [task_id_to_result[t.task_id] for t in all_tasks if t.task_id in task_id_to_result]
        
        return ordered_results
