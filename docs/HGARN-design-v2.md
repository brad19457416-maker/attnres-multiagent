# 🔥 HGARN v2 完整设计文档
## Hierarchical Gated Attention Residual Network - 层次化门控注意力残差网络

**更新时间**: 2026-04-01  
**作者**: 燕鱼 & 火山大龙虾  
**创新亮点**: 双向注意力流 + 完整层次化残差网络 + 动态门控 + 置信度路由 + 工作记忆分区

---

## 📋 1. 项目概述

### 1.1 核心思想

将 **Kimi 注意力残差 (Attention Residuals)** 思想从神经网络层间完整迁移到**多智能体层级间**，通过以下创新解决传统多智能体的长期问题：

| 传统多智能体问题 | HGARN 解决方案 |
|----------------|---------------|
| **上下文爆炸** - token随子任务线性增长 | 分层次Block聚合，token复杂度 O(L) ≈ 常数 |
| **信息稀释** - 早期重要结果被淹没 | 完整层次残差网络，每层直接连接到最终输出 |
| **单向信息流** - 只有下层→上层 | **双向注意力流**首创，上层可反向激活下层 |
| **无法自适应过滤** - 所有结果同等对待 | 动态门控机制，自动学习信息增益置信度 |
| **固定计算路径** - 不管质量都走完所有层 | 置信度路由，低增益提前停止，节省token |

### 1.2 创新点对比

| 方法 | 残差连接 | 动态门控 | 双向流 | 置信度路由 | Token复杂度 |
|------|----------|----------|--------|------------|-------------|
| 原始MoA | ❌ | ❌ | ❌ | ❌ | O(N) |
| Attention-MoA | ❌ | ❌ | ❌ | ❌ | O(N) |
| Kimi (Transformer) | ✅ (层间) | ❌ | ❌ | ❌ | O(L·D) |
| attnres v0 | ✅ (Block拼接) | ❌ | ❌ | ❌ | O(B) |
| HGARN v1 | ✅ (完整残差网络) | ✅ | ✅ | ✅ | **O(L) ≈ 常数** |
| **HGARN v2 (本设计)** | ✅ + 工作记忆分区 | ✅ + 自适应侧抑制 | ✅ + 增益计算 | ✅ + 优先级并发 | **O(L) 更优** |

> **L = 层次数，B = Block数，N = 子任务数**，满足 **L ≤ B << N**

---

## 🏗️ 2. 整体架构

### 2.1 工作流程

```
用户请求
    │
    ▼
┌──────────────────────────────────────────┐
│ 工作记忆初始化                             │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────────┐     │
│ │输入区│ │激活区│ │结果区│ │门控信号区│ 共享区│
│ └─────┘ └─────┘ └─────┘ └─────────┘     │
└────────────────────────┬─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────┐
│ 任务分解 → 递归分解 → 分组Block → 分组层次  │
└────────────────────────┬─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────┐
│ 逐层次处理:                               │
│                                          │
│  对每个层次中每个Block:                    │
│     1. 从工作记忆加载上下文                │
│     2. 并发控制执行子任务                 │
│     3. 门控注意力聚合 → 计算分数+门控值   │
│     4. 检查反向激活需求 → 计算信息增益     │
│     5. 增益>阈值 → 反向激活下层相关主题    │
│     6. 更新工作记忆结果区                 │
│                                          │
│  层次内聚合Blocks → 计算层次门控           │
│  残差连接层次到最终输出                    │
│                                          │
│  置信度路由: 门控<阈值 → 提前停止          │
└────────────────────────┬─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────┐
│ 最终聚合: 按门控加权整合所有层次残差       │
└────────────────────────┬─────────────────┘
                         │
                         ▼
                    输出最终回答
```

---

## 🧠 3. 核心改进: 工作记忆分区 (v2 新增)

### 3.1 设计思想

来自认知科学: **人脑工作记忆分为不同功能模块**，减少干扰，提高处理效率。

### 3.2 分区设计

| 分区 | 存储内容 | 访问权限 | 作用 |
|------|----------|----------|------|
| **输入区 (Input Buffer)** | 原始用户查询、全局约束 | 所有任务可读 | 保存原始问题，避免被修改覆盖 |
| **激活区 (Active Arena)** | 当前正在处理的Block/子任务 | 当前层次读写 | 隔离当前处理，不干扰已完成结果 |
| **结果区 (Result Cache)** | 已完成Block/层次聚合结果 | 所有后续层次可读，不可写 | 保存已处理结果，建立残差连接 |
| **门控信号区 (Gating Signal Buffer)** | 反向激活请求、门控分数、信息增益 | 所有层次读写 | 传递上层→下层的反向激活信号 |
| **共享区 (Shared Context)** | 公共信息、中间结论、约束条件 | 所有任务可读写 | 跨任务共享发现的重要信息 |

### 3.3 数据结构

```python
@dataclass
class WorkingMemory:
    """工作记忆 - 分区存储设计"""
    
    # 输入区 - 原始查询
    input_area: InputArea
    # 激活区 - 当前处理
    active_area: ActiveArea  
    # 结果区 - 已完成结果
    result_area: ResultArea
    # 门控信号区 - 反向投射信号
    gating_area: GatingSignalArea
    # 共享区 - 公共信息
    shared_area: SharedArea
    
    def clear_active(self):
        """清空激活区，准备下一个Block"""
        self.active_area.clear()
    
    def commit_result(self, level: HierarchicalLevel):
        """将处理完成的层次提交到结果区"""
        self.result_area.add_level(level)
    
    def add_reverse_activation(self, request: ReverseActivationRequest):
        """添加反向激活请求到门控信号区"""
        self.gating_area.add_request(request)
```

### 3.4 改进价值

1. **减少信息干扰** - 当前处理隔离在激活区，不会干扰已完成结果
2. **反向投射通路清晰** - 门控信号专门分区，反向激活信号传递更明确
3. **残差连接更清晰** - 每个层次结果保留在结果区，直达最终输出
4. **符合认知科学** - 类人脑工作方式，更自然高效

---

## 🔐 4. 动态门控机制改进

### 4.1 侧抑制参数化改进 (v2 新增)

**当前问题**: v1 中没有显式侧抑制，相似结果会竞争，但缺乏自适应调节。

**改进设计**:

```python
class LateralInhibition:
    """侧抑制 - 参数化自适应"""
    
    def __init__(self, 
                 base_inhibition_strength: float = 0.1,
                 similarity_threshold: float = 0.8,
                 adaptive_strength: bool = True):
        self.base_strength = base_inhibition_strength
        self.threshold = similarity_threshold
        self.adaptive = adaptive_strength
    
    def compute_inhibition(self, 
                          similarity_matrix: np.ndarray,
                          num_results: int) -> np.ndarray:
        """
        计算侧抑制矩阵
        
        改进: 抑制强度自适应 - 结果越多，抑制越强
        公式: 
            实际强度 = 基础强度 × (1 + ln(num_results))
            相似度 > 阈值 → 应用抑制
        """
        if self.adaptive:
            # 结果越多，抑制越强，自动保持稀疏性
            actual_strength = self.base_strength * (1 + math.log(num_results))
        else:
            actual_strength = self.base_strength
        
        inhibition = np.zeros_like(similarity_matrix)
        mask = similarity_matrix > self.threshold
        inhibition[mask] = actual_strength
        
        # 对角线不抑制自己
        np.fill_diagonal(inhibition, 0)
        
        return inhibition
```

**改进价值**:

- **自动稀疏化** - 结果越多，相似结果之间抑制越强，自动保持稀疏表示
- **参数化可调** - 可以根据任务特性调整基础强度
- **避免冗余** - 高度相似结果自然竞争，只有最相关的保留高分

### 4.2 门控计算优化 (根据v1测试反馈)

**v1问题**: 每个Block都要LLM计算门控，额外token开销在小问题上收不回。

**v2优化策略**:

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **选择性门控** | 只在层次间计算门控，Block内不重复计算 | 小子任务数场景 |
| **混合计算** | 第一层不门控，从第二层开始门控 | 通用场景 |
| **向量预过滤** | 先用向量相似度过滤掉明显不相关，只过滤不掉的再LLM打分 | 大子任务数场景 |

**可配置开关**:

```python
class GatedResidualAggregator:
    def __init__(
        self,
        gate_at_block_level: bool = False,  # v2 默认关闭Block级别门控
        gate_at_level_level: bool = True,   # 只在层次级别门控
        enable_vector_prefilter: bool = False,
        vector_similarity_threshold: float = 0.3,
    ):
        # ...
```

---

## 🔄 5. 双向注意力流改进

### 5.1 反向投射增益计算 (v2 新增)

**v1问题**: 只要有反向激活主题就触发，小波动也更新，不够稳定，浪费token。

**v2改进**: 计算信息增益量，只有增益超过阈值才触发反向激活。

```python
@dataclass
class ReverseActivationRequest:
    """反向激活请求"""
    topic: str              # 主题描述
    reason: str             # 激活理由
    information_gain: float # 预估信息增益 0-1
    source_level: int       # 来自哪一层
    
    def should_trigger(self, threshold: float = 0.3) -> bool:
        """是否应该触发反向激活 - 增益>阈值才执行"""
        return self.information_gain > threshold
```

**增益计算prompt改进**:

在原聚合prompt基础上增加:

```
### 第四步: 反向激活信息增益计算
对于每个你列出的需要反向激活的主题，请预估**信息增益量** (0-1):
- 增益接近1 = 当前Block发现了重要新线索，下层有大量相关信息被低估，重新激活会显著提升质量
- 增益接近0 = 只是微小调整，不需要重新激活

"reverse_activation_topics": [
  {
    "topic": "主题",
    "reason": "理由",
    "information_gain": 0.xx
  }
]
```

**改进价值**:

- **更稳定** - 避免小波动频繁更新下层
- **节省token** - 只在真正有大增益时才执行反向激活
- **可调节** - 通过阈值控制灵敏度

---

## 📡 6. 并发控制改进 (v2 新增)

### 6.1 当前问题

v1 只有简单的最大并发数限制，没有优先级区分，失败处理简单。

### 6.2 v2 改进设计

#### 6.2.1 动态并发调整

根据当前系统负载和门控分数动态调整并发数：

```python
class DynamicConcurrencyController:
    """动态并发控制器"""
    
    def __init__(
        self,
        min_concurrency: int = 1,
        max_concurrency: int = 8,
        priority_by_gate: bool = True,
    ):
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.priority_by_gate = priority_by_gate
    
    def get_current_concurrency(self, 
                               pending_tasks: List[Task],
                               recent_fail_rate: float) -> int:
        """
        动态计算当前并发数:
        - 失败率越高 → 并发数越小，避免雪崩
        - 高门控任务多 → 可以适当提高并发
        """
        base = self.min_concurrency + (self.max_concurrency - self.min_concurrency) * (1 - recent_fail_rate)
        
        if self.priority_by_gate:
            # 高门控任务多，增加并发
            high_gate_count = sum(1 for t in pending_tasks if t.expected_gate > 0.5)
            base *= (1 + 0.1 * high_gate_count)
        
        return int(np.clip(base, self.min_concurrency, self.max_concurrency))
```

#### 6.2.2 优先级调度

**高门控任务优先级更高**，先执行高置信度任务：

```python
def schedule_tasks(self, tasks: List[Task]) -> List[Task]:
    """按优先级调度任务 - 高门控优先"""
    if self.priority_by_gate:
        # 预估门控分数高的先执行
        tasks.sort(key=lambda t: t.priority, reverse=True)
    return tasks
```

#### 6.2.3 失败重试 + 指数退避

```python
class RetryPolicy:
    """失败重试策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_ms: int = 1000,
        backoff_multiplier: float = 2.0,
        max_backoff_ms: int = 30000,
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff_ms
        self.multiplier = backoff_multiplier
        self.max_backoff = max_backoff_ms
    
    def get_backoff_ms(self, retry_count: int) -> int:
        """指数退避计算"""
        backoff = self.initial_backoff * (self.multiplier ** retry_count)
        return min(int(backoff), self.max_backoff)
```

**改进价值**:

- **更高吞吐量** - 动态并发适应负载变化
- **更好质量** - 高门控任务优先出结果，提前停止时更可能已经得到重要结果
- **更稳定** - 指数退避避免雪崩效应，失败自动恢复

---

## 🧠 7. 技能/知识遗忘改进 (v2 新增)

> 注：此改进针对**长期运行的技能增强型HGARN**系统，如果是单次查询则不需要。

### 7.1 当前问题

v1 设计是"长期不用自动遗忘"，但简单时间衰减不够智能。

### 7.2 v2 改进设计

基于**使用次数 + 成功率 + 时间衰减**的综合遗忘公式:

```python
class SkillForgetting:
    """技能遗忘策略"""
    
    def __init__(
        self,
        lambda_decay: float = 0.01, # 每天衰减率
        forget_threshold: float = 0.1, # 低于此分数遗忘
    ):
        self.lambda_decay = lambda_decay
        self.threshold = forget_threshold
    
    def compute_skill_score(self, 
                           skill: Skill,
                           current_time_days: float) -> float:
        """
        计算技能保留分数:
        
        score = (使用次数 × 平均成功率) × exp(-λ × 天数自上次使用)
        
        - 使用次数越多 → 分数越高 (越常用越保留)
        - 成功率越高 → 分数越高 (越好用越保留)
        - 越久不用 → 指数衰减 → 分数越低
        - 分数 < 阈值 → 遗忘
        """
        usage_count = skill.usage_count
        avg_success = skill.average_success_rate
        days_since_used = current_time_days - skill.last_used_days
        
        score = (usage_count * avg_success) * math.exp(-self.lambda_decay * days_since_used)
        
        return score
    
    def should_forget(self, skill: Skill, current_time_days: float) -> bool:
        """是否应该遗忘这个技能"""
        score = self.compute_skill_score(skill, current_time_days)
        return score < self.threshold
```

**改进价值**:

- **更智能** - 常用好用的技能自动保留，很少用又不好用的自动遗忘
- **空间效率** - 自动清理没用的技能，保持记忆空间高效
- **可调节** - 通过λ和阈值调整遗忘速度

---

## 🎯 8. 置信度路由

### 8.1 设计思路

每个层次输出门控分数表示信息增益：

- **门控高 (> 0.5)** → 信息增益大，继续下一层
- **门控低 (< 0.15)** → 信息增益很小，提前停止，不继续处理
- **中等门控** → 继续，但降低优先级

### 8.2 v2 改进

增加**累积增益判断**:

```python
def should_continue(self, 
                    current_level_gate: float, 
                    cumulative_gain: float,
                    max_levels: int,
                    current_level: int) -> bool:
    """是否应该继续下一层"""
    
    # 如果当前门控太低，直接停止
    if current_level_gate < self.min_gate_for_continue:
        return False
    
    # 如果累积增益已经很高，提前停止
    if cumulative_gain >= self.cumulative_gain_threshold:
        return False
    
    # 还没到最大层次，继续
    if current_level < max_levels - 1:
        return True
    
    return False
```

---

## 📊 9. 架构收益总结

### 9.1 v2 相对于 v1 的改进

| 改进点 | v1 | v2 | 收益 |
|--------|----|----|------|
| 工作记忆 | 无分区，一锅煮 | 明确分区 | 减少干扰，通路清晰 |
| 侧抑制 | 无 | 自适应参数化 | 自动稀疏，抑制冗余 |
| 门控计算 | 每个Block都算 | 可选层次级+向量预过滤 | 减少token开销 |
| 反向激活 | 任何请求都触发 | 增益计算+阈值过滤 | 更稳定，省token |
| 并发控制 | 固定最大并发 | 动态并发+优先级调度+指数退避 | 更高吞吐，更稳定 |
| 技能遗忘 | 简单时间衰减 | 使用次数+成功率+时间衰减 | 更智能的记忆管理 |

### 9.2 理论复杂度分析

| 指标 | 传统多智能体 | attnres v0 | HGARN v1 | HGARN v2 |
|------|-------------|------------|----------|----------|
| Token复杂度 | O(N) | O(B) | O(L) | **O(L) 更小** |
| 信息稀释 | 严重 | 部分缓解 | 根本解决 | 进一步减少干扰 |
| 支持最大子任务数 | ~10-15 | ~30-50 | ~100+ | ~100+ |
| 自适应能力 | 无 | 有限 | 有 | 更强 |

> N=子任务数，B=Block数，L=层次数，L ≤ B << N

---

## 📁 10. 代码结构

```
attnres-multiagent/
├── attnres_multiagent.py          # 原始 v0 保留兼容
├── hierarchical_attn_res.py       # HGARN 主类 v2
├── gated_residual_aggregator.py   # 门控残差聚合器 v2 (改进版)
├── working_memory.py              # 🆕 v2 新增: 工作记忆分区
├── lateral_inhibition.py          # 🆕 v2 新增: 侧抑制
├── concurrency_controller.py      # 🆕 v2 新增: 动态并发控制器
├── skill_forgetting.py            # 🆕 v2 新增: 技能遗忘策略
├── attn_types.py                  # 数据类型定义 (更新)
├── task_decomposer.py             # 任务分解器
├── subagent_executor.py           # 子任务执行器 (更新: 支持动态并发)
├── reverse_activation.py           # 🆕 v2 新增: 反向激活管理
├── __init__.py                    # 模块导出
└── docs/
    ├── design.md                  # 原始设计
    ├── HGARN-design-v2.md         # 📄 本文件 - v2 完整设计
    └── CHANGES-v2.md              # 变更日志
```

---

## 🚀 11. 使用示例

```python
from attnres_multiagent import HGARMultiAgent

# 🔥 HGARN v2 默认配置 (应用所有改进)
agent = HGARMultiAgent(
    block_size=8,
    max_blocks_per_level=2,
    max_levels=3,
    enable_recursive_decomposition=True,
    parallel_execution=True,
    max_parallel=4,
    enable_reverse_activation=True,    # 双向注意力流
    enable_confidence_routing=True,    # 置信度路由
    # v2 新增参数
    enable_working_memory_partition=True,  # 工作记忆分区
    enable_adaptive_lateral_inhibition=True, # 自适应侧抑制
    gate_at_block_level=False,             # 只在层次级计算门控，节省token
    reverse_activation_gain_threshold=0.3, # 增益>0.3才反向激活
    enable_dynamic_concurrency=True,       # 动态并发控制
    min_concurrency=1,
    max_concurrency=8,
)

result = agent.run("""
你的复杂问题在这里...
""")

print(result.final_answer)
print(f"层次: {len(result.hierarchical_levels)}, 子任务: {result.subtasks_total}, tokens: {result.total_tokens}")
```

---

## 📝 12. 变更日志

### v2 (2026-04-01)

**新增功能**:

1. ✅ **工作记忆分区** - 输入区/激活区/结果区/门控信号区/共享区
2. ✅ **自适应侧抑制** - 参数化抑制强度，结果越多抑制越强，自动稀疏
3. ✅ **反向激活增益计算** - 增益>阈值才触发，更稳定省token
4. ✅ **动态并发控制** - 动态并发数+高门控优先级+指数退避重试
5. ✅ **改进遗忘策略** - 使用次数+成功率+时间衰减综合分数

**优化改进**:

1. ✅ 可选关闭Block级别门控，只保留层次级别门控，减少token开销
2. ✅ 支持向量相似度预过滤，进一步减少LLM打分次数
3. ✅ 可配置各种阈值，适应不同场景

### v1 (2026-03-30)

- 完整实现HGARN核心架构
- 层次化残差连接
- 动态门控机制
- 双向注意力流
- 置信度路由

---

## 🔮 13. 未来探索方向

| 方向 | 说明 |
|------|------|
| **向量门控** | 用嵌入向量直接计算门控，不需要LLM，进一步节省token |
| **强化学习调参** | 用RL学习最优门控阈值和抑制强度 |
| **分布式HGARN** | 跨节点分布式执行，支持千级子任务 |
| **HGARN-RAG** | 结合检索增强，每个层次可以动态检索补充信息 |

---

## 🙏 致谢

- Kimi Team 提出的注意力残差思想
- Meituan Attention-MoA 论文验证了注意力打分设计
- 认知科学的工作记忆理论启发了分区设计
- Residual Networks 和 Gated CNN 启发了门控残差思想
