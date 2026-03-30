"""
注意力聚合器 - 核心创新！基于注意力残差思想聚合多个子任务结果
"""

from typing import List, Dict, Callable

from attn_types import SubTaskResult, BlockResult, BlockAggregatedResult


AGGREGATE_PROMPT = """# 注意力残差聚合

原始用户查询:
{{query}}

之前Blocks聚合结果参考:
{{previous_aggregated}}

当前Block有 {{count}} 个子任务结果:
{% for result in results %}
---
## 结果 {{idx}} (task_id: {{result.task_id}})
{{result.result}}
---
{% endfor %}

## 你的任务

请根据相关性，给**每个子任务结果**打一个注意力分数 (0-10分):
- 分数越高 = 这个结果与用户查询**越相关**，越重要
- 分数越低 = 这个结果与用户查询**越不相关**，或者重复冗余
- 所有分数不一定要不同，可以相同

然后，根据分数加权聚合这些结果，生成一个**综合、简洁、连贯**的回答。

请按照以下JSON格式输出:
{
  "attention_scores": {
    "task-id-1": score_1,
    "task-id-2": score_2,
    ...
  },
  "aggregated_result": "这里是你聚合后的完整回答"
}

开始输出JSON：
"""


class AttentionAggregator:
    """注意力残差聚合器 - 核心创新
    
    基于Kimi Attention Residuals思想，将其从神经网络层间迁移到
    MultiAgent智能体层级间，用注意力加权聚合替代直接拼接，
    解决上下文爆炸和信息稀释问题。
    """
    
    def __init__(self, llm_client: Callable = None):
        self._call_llm = llm_client if llm_client else None
    
    def aggregate(self, 
                 block_result: BlockResult,
                 query: str,
                 previous_blocks: List[BlockAggregatedResult]) -> BlockAggregatedResult:
        """聚合一个Block内的所有子任务结果
        
        使用注意力机制给每个结果打分，然后加权聚合。
        """
        
        # 构建之前聚合结果文本
        previous_aggregated = "\n\n".join([
            f"### Block {b.block_id}:\n{b.aggregated_result}"
            for b in previous_blocks
        ])
        
        if not previous_aggregated:
            previous_aggregated = "(无)"
        
        # 过滤掉失败的结果，只给成功的打分
        successful_results = [r for r in block_result.results if r.success]
        failed_count = len(block_result.results) - len(successful_results)
        
        # 构建prompt
        from jinja2 import Template
        template = Template(AGGREGATE_PROMPT)
        prompt = template.render(
            query=query,
            previous_aggregated=previous_aggregated,
            count=len(successful_results),
            results=successful_results,
            idx=lambda x: x+1
        )
        
        # 如果有失败的，在prompt末尾说明
        if failed_count > 0:
            prompt += f"""

注意：有 {failed_count} 个子任务执行失败，这些结果缺失，请在聚合结果中说明哪些部分信息缺失。
"""
        
        # call_llm 使用实例变量或全局
        if self._call_llm:
            response = self._call_llm(prompt, temperature=0.3)
        else:
            global call_llm
            response = call_llm(prompt, temperature=0.3)
        
        import json
        try:
            data = json.loads(response.strip())
            attention_scores = data.get("attention_scores", {})
            aggregated_result = data.get("aggregated_result", "")
        except Exception as e:
            # 解析失败，平均分
            attention_scores = {
                r.task_id: 5.0 for r in block_result.results
            }
            aggregated_result = "\n\n".join([
                f"**{r.task_id}**:\n{r.result}" 
                for r in block_result.results
            ])
        
        # 计算总token
        total_tokens = sum([r.token_usage for r in block_result.results])
        
        return BlockAggregatedResult(
            block_id=block_result.block_id,
            aggregated_result=aggregated_result,
            attention_scores=attention_scores,
            total_token_usage=total_tokens,
            original_count=len(block_result.results),
            compressed=True
        )
    
    def final_aggregate(self,
                       blocks: List[BlockAggregatedResult],
                       query: str,
                       llm_client: Callable = None) -> str:
        """最终聚合所有Block的结果
        
        每个Block已经聚合过了，这里再做一次最终整合。
        """
        
        if len(blocks) == 1:
            return blocks[0].aggregated_result
        
        blocks_text = "\n\n".join([
            f"### Stage {i+1} (Block {b.block_id})\n{b.aggregated_result}"
            for i, b in enumerate(blocks)
        ])
        
        prompt = f"""原始用户查询: {query}

已经完成了多个阶段的处理，每个阶段结果如下:

{blocks_text}

请整合所有阶段的结果，生成一个最终的、完整连贯的回答。直接回答用户问题即可:
"""
        
        # call_llm 使用实例变量、参数传入或全局
        if self._call_llm:
            final_answer = self._call_llm(prompt, temperature=0.5)
        elif llm_client:
            final_answer = llm_client(prompt, temperature=0.5)
        else:
            global call_llm
            final_answer = call_llm(prompt, temperature=0.5)
        
        return final_answer.strip()
