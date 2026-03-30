"""
🔥 门控残差聚合器 - 创新核心！层次化门控注意力残差聚合
===

这是我们相对于原始attnres-multiagent的核心创新：
1. **层次化残差连接** —— 每一层/每个Block都建立到最终输出的残差连接
2. **动态门控机制** —— 学习门控值决定哪些信息保留，哪些遗忘
3. **双向注意力流** —— 上层结果可以反向激活下层关键信息
4. **置信度路由** —— 低置信度结果自动触发重新执行

论文思想来源:
- Kimi Attention Residuals: 残注意力残差基础
- Attention-MoA: 显式打分设计
- Gated Residual Networks: 门控机制启发
- Router Network: 动态路由思想
"""

from typing import List, Dict, Callable, Optional
from dataclasses import dataclass

from .attn_types import (
    SubTaskResult, 
    BlockResult, 
    BlockAggregatedResult,
    HierarchicalLevel
)


GATED_AGGREGATE_PROMPT = """# 门控注意力残差聚合

## 上下文信息

原始用户查询:
```
{{query}}
```

{% if previous_levels %}
前面各层聚合结果 (带门控分数):
{% for level in previous_levels %}
### Level {{level.level_id}} (门控分数: {{ "%.2f"|format(level.gate_score) }})
{{level.aggregated}}
{% endfor %}
{% endif %}

## 当前Block需要聚合的结果

当前Block有 {{count}} 个子任务结果:
{% for result in results %}
---
## 结果 {{idx}} (task_id: {{result.task_id}})
**结果内容**:
{{result.result}}
---
{% endfor %}

## 你的任务

### 第一步: 给每个子任务打分 (0-10分)
- **分数越高** = 结果与查询**高度相关，质量好，重要**
- **分数越低** = 结果**不相关、重复、质量差**

### 第二步: 计算当前Block的整体门控值 (0.0 - 1.0)
- **门控值越高** = 这个Block包含较多重要新信息，应该保留更多
- **门控值越低** = 这个Block信息增益很小，大部分是重复，可以压缩过滤
- **计算方式**: 门控值 = avg(注意力分数) / 10.0

### 第三步: 聚合当前Block结果
根据注意力分数加权聚合，生成一份**简洁、连贯、信息密度高**的聚合结果。
不重要、不相关的内容要大胆删掉或者简写，只保留重要信息。

### 第四步: 残差信息交互 (创新!)
请给出**反向激活提示** —— 基于当前Block的新信息，你认为前面哪些层次/Block的信息需要被**重新强调**？
列出需要重新激活的主题和理由。

请按照以下JSON格式输出:
{
  "attention_scores": {
    "task-id-1": score_1,
    "task-id-2": score_2,
    ...
  },
  "gate_value": 0.xx,
  "aggregated_result": "这里是你聚合后的完整回答",
  "reverse_activation_topics": [
    {
      "topic": "需要重新激活的主题",
      "reason": "为什么需要重新激活"
    }
  ]
}

开始输出JSON：
"""


HIERARCHICAL_FINAL_AGGREGATE_PROMPT = """# 层次化门控注意力残差 - 最终聚合

## 原始用户查询:
{{query}}

## 各层处理结果 (带门控值，门控值越高越重要):

{% for level in levels %}
---
### Level {{level.level_id}} (门控 = {{ "%.3f"|format(level.gate_score) }})
{{level.aggregated}}
---
{% endfor %}

## 你的任务

所有层次都已经处理完成，每个层次都通过**门控残差连接**直接连接到最终输出。
请你:

1. **整合所有层次**，根据门控值加权，门控高的信息保留完整，门控低的信息简要带过
2. **解决冲突** —— 如果不同层次有矛盾信息，采信门控值高、更新的层次
3. **生成最终回答** —— 完整、连贯、直接回答用户问题，保留所有重要信息，压缩冗余

直接输出最终回答即可，不需要JSON:
"""


REVIVE_ACTIVATION_PROMPT = """# 反向激活下层信息 (残差连接交互)

## 原始用户查询:
{{query}}

## 当前新发现的信息提示需要重新激活以下主题:
{% for topic in topics %}
- **主题**: {{topic.topic}}
- **理由**: {{topic.reason}}
{% endfor %}

## 之前各层聚合结果:
{% for level in previous_levels %}
### Level {{level.level_id}}:
{{level.aggregated}}
{% endfor %}

## 你的任务

根据当前新发现，重新激活提取下层中与这些主题相关的重要信息，重新聚合浓缩。
输出重新聚合后的浓缩结果:
"""


class GatedResidualAggregator:
    """🔥 门控残差聚合器 - 核心创新
    
    实现层次化门控注意力残差聚合:
    1. 每个Block输出都带门控值，表示信息增益
    2. 每个Block都建立残差连接到最终输出
    3. 支持反向激活，上层结果可以唤醒下层关键信息
    4. 最终聚合按门控值加权整合所有残差
    """
    
    def __init__(self, llm_client: Callable = None):
        self._call_llm = llm_client if llm_client else None
    
    def aggregate_block(self, 
                       block_result: BlockResult,
                       query: str,
                       previous_levels: List[HierarchicalLevel]) -> BlockAggregatedResult:
        """聚合一个Block，计算门控值和反向激活主题
        
        Args:
            block_result: Block执行结果
            query: 原始用户查询
            previous_levels: 前面已经处理完的层次
            
        Returns:
            BlockAggregatedResult: 包含注意力分数、聚合结果、门控值
        """
        from jinja2 import Template
        
        # 过滤掉失败的结果
        successful_results = [r for r in block_result.results if r.success]
        failed_count = len(block_result.results) - len(successful_results)
        
        # 构建prompt
        prev_levels_text = []
        for lv in previous_levels:
            prev_levels_text.append({
                "level_id": lv.level_id,
                "aggregated": lv.aggregated,
                "gate_score": lv.gate_score
            })
        
        template = Template(GATED_AGGREGATE_PROMPT)
        prompt = template.render(
            query=query,
            previous_levels=prev_levels_text,
            count=len(successful_results),
            results=successful_results,
            idx=lambda x: x+1
        )
        
        if failed_count > 0:
            prompt += f"""

注意：有 {failed_count} 个子任务执行失败，这些结果缺失。
"""
        
        # 调用LLM
        if self._call_llm:
            response = self._call_llm(prompt, temperature=0.3)
        else:
            global call_llm
            response = call_llm(prompt, temperature=0.3)
        
        import json
        try:
            data = json.loads(response.strip())
            attention_scores = data.get("attention_scores", {})
            gate_value = data.get("gate_value", 0.5)
            aggregated_result = data.get("aggregated_result", "")
            # reverse_activation = data.get("reverse_activation_topics", [])
        except Exception as e:
            # 解析失败，平均分
            attention_scores = {r.task_id: 5.0 for r in block_result.results}
            gate_value = 0.5
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
            compressed=True,
            gate_value=gate_value,
            residual_connection=True
        )
    
    def aggregate_level(self,
                       blocks: List[BlockAggregatedResult],
                       level_id: int,
                       query: str,
                       previous_levels: List[HierarchicalLevel]) -> HierarchicalLevel:
        """聚合一个层次内的多个Block，生成层次聚合结果
        
        一个层次可以包含多个Block，聚合后作为一个层次加入残差网络。
        """
        # 如果只有一个Block，直接用它的结果
        if len(blocks) == 1:
            return HierarchicalLevel(
                level_id=level_id,
                blocks=blocks,
                aggregated=blocks[0].aggregated_result,
                gate_score=blocks[0].gate_value
            )
        
        # 多个Block需要再聚合一次
        blocks_text = "\n\n".join([
            f"### Block {b.block_id} (gate={b.gate_value:.2f})\n{b.aggregated_result}"
            for b in blocks
        ])
        
        prompt = f"""# 层次内聚合

原始查询: {query}

当前层次包含多个Block，请聚合它们：

{blocks_text}

请生成一份综合的聚合结果，保留所有重要信息，压缩冗余:
"""
        
        if self._call_llm:
            result = self._call_llm(prompt, temperature=0.3)
        else:
            global call_llm
            result = call_llm(prompt, temperature=0.3)
        
        # 计算平均门控
        avg_gate = sum([b.gate_value for b in blocks]) / len(blocks)
        
        return HierarchicalLevel(
            level_id=level_id,
            blocks=blocks,
            aggregated=result.strip(),
            gate_score=avg_gate
        )
    
    def reverse_activate_lower(self,
                             topics: List[dict],
                             query: str,
                             previous_levels: List[HierarchicalLevel]) -> str:
        """反向激活 —— 上层结果发现需要重新强调下层信息，重新聚合
        
        这是**双向注意力流**的关键创新：
        信息不仅从下层->上层流动，还可以上层->下层反向流动，
        上层发现新线索，可以唤醒下层相关信息重新强调。
        """
        from jinja2 import Template
        
        template = Template(REVIVE_ACTIVATION_PROMPT)
        prompt = template.render(
            query=query,
            topics=topics,
            previous_levels=previous_levels
        )
        
        if self._call_llm:
            result = self._call_llm(prompt, temperature=0.5)
        else:
            global call_llm
            result = call_llm(prompt, temperature=0.5)
        
        return result.strip()
    
    def final_aggregate(self,
                       levels: List[HierarchicalLevel],
                       query: str) -> str:
        """最终聚合 —— 整合所有层次的门控残差连接，输出最终回答
        
        每个层次都通过残差连接直接到最终输出，按门控值加权整合。
        这就是我们的核心创新 —— **层次化门控注意力残差网络**。
        """
        from jinja2 import Template
        
        template = Template(HIERARCHICAL_FINAL_AGGREGATE_PROMPT)
        prompt = template.render(
            query=query,
            levels=levels
        )
        
        if self._call_llm:
            final_answer = self._call_llm(prompt, temperature=0.5)
        else:
            global call_llm
            final_answer = call_llm(prompt, temperature=0.5)
        
        return final_answer.strip()
