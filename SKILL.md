# attnres-multiagent - 注意力残差多智能体框架

> 融合最新三大技术创新：**Kimi注意力残差** + **DeerFlow 2.0 Skill架构** + **Attention-MoA**，
> 通过Block注意力聚合解决**上下文爆炸**和**信息稀释**问题，
> 让多智能体系统支持更多子任务，保留更多重要信息，token效率提升数倍。

## 🎯 核心创新

| 问题 | 传统多智能体 | attnres-multiagent |
|------|--------------|-------------------|
| **上下文爆炸** | token随子任务数线性增长，很快占满窗口 | Block聚合 → token O(块数) ≈ 常数 |
| **信息稀释** | 早期重要结果被后续结果淹没 | 动态注意力加权 → 重要信息保留高权重 |
| **最大子任务数** | ~10-15 | ~**几十上百** |

**思想来源：**
- **注意力残差**: 《Attention Residuals》Kimi Team, 2026-03
- **DeerFlow 2.0**: ByteDance, 2026-02 - Skill模块化思想
- **Attention-MoA**: Attention-MoA, Meituan, 2026-01 - 多智能体注意力聚合验证

## 🚀 快速开始

```python
from attnres_multiagent import AttnResMultiAgent

# 初始化
agent = AttnResMultiAgent(
    block_size=8,      # 每个Block最多多少个子任务
    max_blocks=3,      # 最多多少个Block
    adaptive_early_stop=True
)

# 运行
result = agent.run("""
你的复杂问题这里放
""")

# 输出结果
print(result.final_answer)
print(f"处理了 {result.blocks_processed} 个Block")
print(f"分解了 {result.subtasks_count} 个子任务")
print(f"消耗 {result.total_tokens} tokens")
```

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `block_size` | 8 | 每个Block最多容纳多少个子任务 |
| `max_blocks` | 3 | 最多允许多少个Block（控制最大深度） |
| `adaptive_early_stop` | True | 是否检查收敛提前停止，如果最后一个Block平均分低于3分就提前停止 |
| `parallel_execution` | False | 是否并行执行可并行子任务，最多同时4个任务 |
| `attn_score_model` | "same" | 用哪个模型计算注意力分数 |
| `enable_recursive_decomposition` | False | 是否启用递归分层分解，允许子任务继续分解 |
| `max_recursion_depth` | 3 | 最大递归分解深度 |

## 🏗️ 架构设计

```
用户请求
    │
    ▼
┌──────────────────────────────────┐
│  主Agent: 任务分解               │
│  将查询分解为多个子任务          │
└────────────────┬─────────────────┘
                 │
         ┌───────┼───────┐
         ▼       ▼       ▼
      ┌──────┐┌──────┐┌──────┐
      │子Agent││子Agent││子Agent│   并行/顺序执行
      └───┬──┘└───┬──┘└───┬──┘
          │       │       │
          ▼       ▼       ▼
       结果1    结果2    结果3
          └───────┼───────┘
                  │
                  ▼
┌──────────────────────────────────┐
│   注意力残差聚合 (Block-AttnRes)  │
│  - 给每个结果计算注意力分数        │
│  - 加权聚合 → 压缩token            │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│   判断是否继续下一个Block？       │
└────────────────┬─────────────────┘
                 │
                 ▼
      聚合所有Block → 输出最终回答
```

### 核心创新：Block注意力残差聚合

将Kimi在Transformer中提出的思想**迁移到多智能体层级**：

| Kimi (Transformer) | attnres-multiagent (MultiAgent) |
|-------------------|--------------------------------|
| 每层关注前面所有层输出 | 每个Block关注前面所有Block输出 |
| 固定加法 → 注意力加权 | 直接拼接 → 注意力聚合 |
| 解决层间信息稀释 | 解决层级间信息稀释 |
| Block分块减少内存 | Block分块控制token增长 |

## 📊 预期收益

| 指标 | 传统拼接 | Attention-MoA | attnres-multiagent |
|------|----------|---------------|-------------------|
| Token复杂度 | O(N) | O(L) | **O(B) ≈ 常数** |
| 信息稀释 | 严重 | 部分缓解 | **Block注意力残差更好缓解** |
| 最大子任务数 | ~10-15 | ~20-30 | **~几十上百** |

## 🛠️ 开发状态

- [x] **MVP版本** - 基础架构完成，单层Block注意力聚合
- [x] **向量快速筛选优化** - top-k相似度预筛选，减少token用量
- [x] **递归分层分解** - 允许子任务进一步分解，最大深度可配置
- [x] **自适应提前停止** - 根据注意力分数判断收敛，提前停止减少token
- [x] **并行执行优化** - 多个可并行子任务并行执行，加速处理
- [x] **DeerFlow Skill格式完全兼容** - 标准OpenClaw Skill封装

## ✅ 全部功能已开发完成

**attnres-multiagent v1.0** 包含完整功能：

| 功能 | 状态 |
|------|------|
| 基础Block注意力残差聚合 | ✅ |
| Attention-MoA打分设计 | ✅ |
| 任务自动分解 | ✅ |
| Block分组控token | ✅ |
| 递归分层分解 | ✅ |
| 自适应提前停止 | ✅ |
| 并行执行优化 | ✅ |
| 向量相似度预筛选 | ✅ |
| 最终整合输出 | ✅ |

## 📝 设计方案

完整设计方案保存在：
`/root/.openclaw/workspace/projects/attnres-multiagent/design.md`

## 🎯 适用场景

- **深度研究任务** - 需要分解很多子问题
- **复杂分析** - 多角度分析，需要保留各角度信息
- **多视角评审** - 多个子Agent从不同角度分析，需要聚合
- **增量式问题解决** - 分阶段处理，每阶段需要保留之前结论

## 👨‍💻 对于 Agent 开发者：如何在你的 Agent 中使用

### 什么时候应该用 AM？

| 问题规模 | 是否用 AM | 原因 |
|----------|----------|------|
| 简单问题（1-3个子问题） | ❌ 不用 | 直接回答更快 |
| 中等问题（4-8个子问题） | ✅ 推荐用 | 控制token，避免稀释 |
| 复杂问题（8+个子问题） | ✅ **必须用** | 传统方式一定会token爆炸 |

### 使用决策树

```
你接到一个用户问题
    │
    ▼
问题能不能一句话回答？ → 是 → 直接回答
    │
    ▼
能分解成几个独立子问题？ → 不能 → 直接回答
    │
    ▼
子问题数量 > 3 → ✅ 使用 attnres-multiagent！
```

### 代码示例（复制就能用）

```python
# 作为 Agent 开发者，你只需要：
from attnres_multiagent import AttnResMultiAgent

def process_complex_query(user_query: str) -> str:
    # 初始化（启用所有优化）
    agent = AttnResMultiAgent(
        block_size=8,
        max_blocks=3,
        adaptive_early_stop=True,      # 推荐开启
        parallel_execution=True,       # 推荐开启，加速
        enable_recursive_decomposition=True,  # 推荐开启，自动分解复杂子任务
        max_recursion_depth=3
    )
    
    # 直接运行，AM 帮你处理一切：
    # 1. 自动分解任务
    # 2. 递归分解（如果启用）
    # 3. 分组分块
    # 4. 执行子任务
    # 5. 注意力聚合压缩token
    # 6. 提前停止（如果收敛）
    # 7. 最终整合输出
    result = agent.run(user_query)
    
    # 返回最终结果给用户
    return result.final_answer
```

### 最佳实践建议

1. **默认开启所有优化**：`adaptive_early_stop=True`, `parallel_execution=True`, `enable_recursive_decomposition=True`
2. **block_size 保持 8**：这是经验最优值，不要太大（token爆炸）不要太小（增加Block数）
3. **max_blocks 保持 3**：3 个Block足够绝大多数场景，token 可控
4. **max_recursion_depth 保持 3**：3层足够分解到可执行粒度

## 📄 许可证

MIT
