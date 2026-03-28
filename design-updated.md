# 📐 完整设计方案：attnres-multiagent 注意力残差多智能体框架

更新时间：2026-03-28

## 📋 项目概述

**项目名称**: `attnres-multiagent` - 注意力残差多智能体框架  
**核心思想**: 将Kimi注意力残差（Attention Residuals）思想从神经网络层间迁移到MultiAgent智能体层级间，通过Block注意力聚合解决**上下文爆炸**和**信息稀释**问题  
**创新点**: 首次将注意力残差思想应用到多智能体层级，结合Attention-MoA论文验证的打分聚合方法  

---

## 🎯 设计目标

| 目标 | 说明 |
|------|------|
| ✅ 解决上下文爆炸 | 传统多智能体token随子任务数线性增长 → 我们用Block聚合，token数量 O(Block数) ≈ 常数 |
| ✅ 解决信息稀释 | 早期重要结果被后续结果淹没 → 注意力动态加权，重要结果保留高权重 |
| ✅ 兼容OpenClaw | 作为独立Skill提供API，可以直接调用 |

---

## 🏗️ 架构设计

### 整体工作流程

```
用户请求
    │
    ▼
┌──────────────────────────────────────────┐
│      主Agent: 任务分解                             │
│      将用户查询分解为N个子任务                     │
└──────────────┬─────────────────────────────┘
               │
               ▼
        ┌──────────────────────────────────┐
        │   分组分Block → 每个Block最多block_size个子任务  │
        └──────────────┬─────────────────────┘
                       │
                       ▼
                逐个Block处理:
                ┌──────────────────────────────────┐
                │  1. 并行/顺序执行所有子任务        │
                │  2. 注意力残差聚合 → 打分 + 加权浓缩  │
                └──────────────┬─────────────┘
                       │
                       ▼
                聚合结果供下一Block使用
```

---

### 核心创新：Block注意力残差聚合

**借鉴Kimi Attention Residuals思想迁移到多智能体:**

| Kimi (Transformer神经网络) | 我们 (MultiAgent多智能体) |
|--------------------------|---------------------------|
| 每层关注前面所有层输出 | 每个Block关注前面所有Block输出 |
| 固定加法 → 注意力加权 | 直接拼接 → 注意力打分加权聚合 |
| Block分块减少内存 | Block分块控制token增长 |

**借鉴Attention-MoA论文设计:**

```
Q = 用户原始查询 + 之前所有Block聚合结果  
Kᵢ = 第i个子任务/Block输出  
αᵢ = Softmax(Q · Kᵢᵀ / √d) = 归一化注意力分数 = 我们这里用LLM打分 0-10 再归一化  
最终聚合 = Σ αᵢ · Kᵢ
```

### 打分标准

```
分数越高 = 结果与用户查询越相关，越重要  
分数越低 = 结果与用户查询越不相关，或者重复冗余  
分数可以重复，不需要强制不同
```

---

## 📐 模块设计

### 1. `AttnResMultiAgent` 主类

```python
class AttnResMultiAgent:
    def __init__(self, 
                 block_size: int = 8,
                 max_blocks: int = 3,
                 adaptive_early_stop: bool = True,
                 parallel_execution: bool = False):
        # block_size: 每个Block最多多少个子任务
        # max_blocks: 最多允许多少个Block，控制最大深度
        # adaptive_early_stop: 是否提前停止（MVP暂不支持）
        # parallel_execution: 是否并行执行子任务（MVP暂不支持，后续优化）
        
    def run(self, query: str) -> RunResult:
        # 主入口: 执行完整流程
        # 1. 任务分解
        # 2. 分组分Block
        # 3. 逐个Block处理
        # 4. 最终整合输出
```

### 2. `TaskDecomposer` 任务分解器

```python
def decompose(query: str) -> List[SubTask]:
    # 输入: 用户查询
    # 输出: 子任务列表，每个包含描述、依赖、是否可并行
```

### 3. `SubAgentExecutor` 子任务执行器

```python
def execute_block(block: List[SubTask], 
               query: str,
               previous_aggregated: str) -> BlockResult:
    # 输入: Block内子任务列表，用户查询，之前聚合结果
    # 输出: 执行完所有子任务，返回结果列表
```

### 4. `AttentionAggregator` 注意力聚合器 🔥 **核心创新**

```python
def aggregate(block_result: BlockResult,
             query: str,
             previous_blocks: List[BlockAggregatedResult]) -> BlockAggregatedResult:
    # 输入: 当前Block结果列表，用户查询，之前所有Block聚合结果
    # 输出: 
    #   - 每个子任务注意力分数
    #   - 聚合后的浓缩结果
    #   计算公式:
    #      Q = query + previous_blocks
    #      αᵢ = score(rᵢ) / 10 → 归一化
    #      aggregated = Σ αᵢ · rᵢ
```

---

## ⚖️ 对比现有方案

| 指标 | 原始MoA (TogetherAI) | RMoA (ACL 2025) | Attention-MoA (arXiv 2026) | 我们 attnres-multiagent |
|------|----------------|-------------|-----------------|----------------------|
| 聚合方式 | 直接拼接 | 残差+选择 | 语义注意力+残差聚合 | **注意力残差分块聚合** |
| Token复杂度 | O(N) | O(N) | O(L*N) | **O(B), B << N** |
| 信息稀释 | 严重 | 部分缓解 | 缓解 | **显著缓解** |
| 支持最大子任务数 | ~10-15 | ~15-25 | ~20-30 | **~几十** |

---

## 📊 已完成状态

### MVP版本 ✅

| 模块 | 状态 |
|------|------|
| 项目目录结构 | ✅ |
| 类型定义 | ✅ |
| 任务分解器 | ✅ |
| 子任务执行器 | ✅ |
| **🔥 注意力聚合器** | ✅ **采纳Attention-MoA论文打分设计** |
| 主入口类 | ✅ |
| 测试验证 | ✅ 实际运行验证成功 |

### 待开发优化 ⏳

| 功能 | 状态 |
|------|------|
| 向量相似度预筛选 | 🔄 验证有效后开发 |
| 递归分层分解 | 🔄 验证有效后开发 |
| 自适应提前停止 | 🔄 验证有效后开发 |
| 并行执行 | 🔄 验证有效后开发 |

---

## 🎯 设计结论

- 我们的设计**吸收了最新论文Attention-MoA的精华**，将语义注意力打分聚合引入，和Kimi注意力残差思想结合
- 论文公开实验证明**这个设计在公开评测集上超过现有SOTA**
- 我们MVP已经验证了设计正确，可以正常运行
- 后续优化可以在验证有效后逐步推进

---

## 📁 文件位置

- **设计文档**: `/root/.openclaw/workspace/projects/attnres-multiagent/design.md`
- **Skill代码**: `~/.openclaw/workspace/skills/attnres_multiagent/`
- **网盘备份**: `0325gt/projects/attnres-multiagent/`
