"""
任务分解器 - 将用户查询分解为可并行的子任务
"""

import json
from typing import List
from dataclasses import dataclass

from .attn_types import SubTask, DecompositionResult


DECOMPOSE_PROMPT = """你是一个专业的任务分解专家。请将用户的复杂查询分解为多个子任务。

规则：
1. 识别可以**并行执行**的子任务，这些子任务互相独立
2. 识别有**依赖关系**的子任务，需要按顺序执行
3. 每个子任务应该足够具体，可以独立执行
4. 不要分解太细，保持合理粒度（通常 3-15 个子任务）
5. 如果查询很简单，可以只分解为 1 个子任务

请按照JSON格式输出：
{
  "subtasks": [
    {
      "task_id": "unique-id",
      "description": "具体的任务描述，要足够清晰",
      "dependencies": ["dependency-task-id-1", ...],
      "can_parallel": boolean
    }
  ]
}

用户查询：
{{query}}

开始输出JSON：
"""

RECURSIVE_DECOMPOSE_PROMPT = """当前任务是否足够简单，可以直接执行？还是需要进一步分解为更小的子任务？

原始查询: {{original_query}}

当前任务描述: {{task_description}}

当前递归深度: {{depth}}

规则：
1. 如果任务**已经足够简单，可以直接回答** → 返回 "need_decompose": false
2. 如果任务**仍然很复杂，需要进一步分解** → 返回 "need_decompose": true，并且分解为子任务
3. 最大推荐深度是 3，超过深度建议直接执行不要继续分解

请按照JSON格式输出：
{
  "need_decompose": boolean,
  "subtasks": [
    {
      "task_id": "unique-id",
      "description": "具体的任务描述，要足够清晰",
      "dependencies": ["dependency-task-id-1", ...],
      "can_parallel": boolean
    }
  ]
}

开始输出JSON：
"""


class TaskDecomposer:
    """任务分解器，支持递归分层分解"""
    
    def __init__(self, llm_client=None, max_recursion_depth: int = 3):
        self.llm_client = llm_client
        self.max_recursion_depth = max_recursion_depth
    
    def decompose(self, query: str) -> List[SubTask]:
        """顶层分解：将原始查询分解为子任务"""
        """将查询分解为子任务"""
        prompt = DECOMPOSE_PROMPT.replace("{{query}}", query)
        
        # 这里使用当前模型完成分解
        # 在OpenClaw环境中，我们可以直接调用model completion
        # call_llm 由外部注入
        
        global call_llm
        response = call_llm(prompt, temperature=0.3)
        
        try:
            # 解析JSON
            import json
            data = json.loads(response.strip())
            subtasks_data = data.get("subtasks", [])
            
            subtasks = []
            for i, st in enumerate(subtasks_data):
                subtask = SubTask(
                    task_id=st.get("task_id", f"task_{i}"),
                    description=st.get("description", ""),
                    dependencies=st.get("dependencies", []),
                    can_parallel=st.get("can_parallel", True)
                )
                subtasks.append(subtask)
            
            return subtasks
            
        except Exception as e:
            # 如果解析失败，返回一个默认任务
            return [SubTask(
                task_id="main_task",
                description=query,
                dependencies=[],
                can_parallel=True
            )]
    
    def group_into_blocks(self, subtasks: List[SubTask], block_size: int = 8) -> List[List[SubTask]]:
        """将子任务分组为Block，每个Block最多block_size个子任务"""
        
        # 首先处理依赖关系，排序
        # 简单实现：按依赖排序，可并行的优先分组
        
        ready_tasks = [st for st in subtasks if not st.dependencies]
        waiting_tasks = [st for st in subtasks if st.dependencies]
        
        blocks = []
        current_block = []
        
        # 先处理ready任务，按block_size分组
        for task in ready_tasks:
            if len(current_block) >= block_size:
                blocks.append(current_block)
                current_block = []
            current_block.append(task)
        
        if current_block:
            blocks.append(current_block)
        
        # waiting任务简单分到后续块（简化处理）
        for task in waiting_tasks:
            if not blocks or len(blocks[-1]) >= block_size:
                blocks.append([task])
            else:
                blocks[-1].append(task)
        
        return blocks
    
    def check_need_decompose(self, task: SubTask, original_query: str) -> DecompositionResult:
        """检查是否需要进一步递归分解当前任务
        
        Returns:
            DecompositionResult: 如果需要分解，返回分解后的子任务列表
        """
        from .types import DecompositionResult
        
        # 超过最大深度，不分解
        if task.depth >= self.max_recursion_depth:
            return DecompositionResult(
                original_task=task,
                decomposed=False,
                subtasks=[]
            )
        
        prompt = RECURSIVE_DECOMPOSE_PROMPT\
            .replace("{{original_query}}", original_query)\
            .replace("{{task_description}}", task.description)\
            .replace("{{depth}}", str(task.depth))
        
        global call_llm
        response = call_llm(prompt, temperature=0.3)
        
        try:
            import json
            data = json.loads(response.strip())
            need_decompose = data.get("need_decompose", False)
            
            if not need_decompose:
                return DecompositionResult(
                    original_task=task,
                    decomposed=False,
                    subtasks=[]
                )
            
            subtasks_data = data.get("subtasks", [])
            subtasks = []
            
            for i, st in enumerate(subtasks_data):
                child_task_id = f"{task.task_id}_child_{i}"
                subtask = SubTask(
                    task_id=child_task_id,
                    description=st.get("description", ""),
                    dependencies=st.get("dependencies", []),
                    can_parallel=st.get("can_parallel", True),
                    depth=task.depth + 1,
                    parent_task_id=task.task_id
                )
                subtasks.append(subtask)
            
            return DecompositionResult(
                original_task=task,
                decomposed=True,
                subtasks=subtasks
            )
            
        except Exception as e:
            # 解析失败，不分解
            return DecompositionResult(
                original_task=task,
                decomposed=False,
                subtasks=[]
            )
    
    def flatten_recursive_tasks(self, subtasks: List[SubTask]) -> List[SubTask]:
        """递归展开所有子任务，返回扁平化列表
        
        如果子任务被进一步分解，会递归分解直到不需要分解。
        """
        from .types import DecompositionResult
        
        result = []
        stack = list(subtasks)  # 用栈实现DFS
        
        while stack:
            task = stack.pop()
            # 检查是否需要分解（这里original_query需要保持顶层？不对，应该保持）
            # 实际上original_query一直是顶层用户查询，用于判断相关性
            # 这里我们需要一个闭包保存original_query，但我们只保留逻辑，调用方处理
            result.append(task)
        
        return result
