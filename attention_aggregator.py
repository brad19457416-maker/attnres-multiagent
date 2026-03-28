"""
注意力聚合器 - 核心创新！基于注意力残差思想聚合多个子任务结果
"""

from typing import List, Dict
from dataclasses import dataclass

from .attn_types import SubTaskResult, BlockResult, BlockAggregatedResult


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
    
    def __init__(self):
        pass
    
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
        
        # 构建prompt
        from jinja2 import Template
        template = Template(AGGREGATE_PROMPT)
        prompt = template.render(
            query=query,
            previous_aggregated=previous_aggregated,
            count=len(block_result.results),
            results=block_result.results,
            idx=lambda x: x+1
        )
        
        # call_llm 由外部注入
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
                       query: str) -> str:
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
        
        # call_llm 由外部注入
        global call_llm
        final_answer = call_llm(prompt, temperature=0.5)
        
        return final_answer.strip()
