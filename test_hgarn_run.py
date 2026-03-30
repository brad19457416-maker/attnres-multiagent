#!/usr/bin/env python3
"""
HGARN v1 测试 - 层次化门控注意力残差网络
直接使用当前环境的call_llm
"""

import sys
import os
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入（当前目录已经在路径中，所有相对导入已经改成直接导入）
from attn_types import (
    SubTask, 
    SubTaskResult, 
    BlockResult, 
    BlockAggregatedResult,
    HierarchicalLevel,
    RunResult
)
from hierarchical_attn_res import HGARMultiAgent

# 测试问题 - 和v0完全一样，保证对比公平
TEST_QUERY = """请帮我全面分析2025-2026年AI大模型领域的发展，包括:

1. 当前主要的技术路线分歧有哪些？比如开源闭源、MoE vs 密集模型、推理优化方向、架构创新等，每个路线详细分析
2. 对比各路线的优缺点，以及目前的性能/成本 trade-off
3. 国内外主要玩家（OpenAI、Anthropic、Google、字节、百度、阿里等）各自布局了哪些路线？
4. 商业化落地进展如何？有哪些成功的商业模式？
5. 预测未来12个月（到2027年3月）的发展趋势，有哪些技术会突破，哪些投资机会值得关注？

请全面深入分析，每个点都要有具体内容。"""


def main():
    print("=" * 70)
    print("🔥 HGARN v1 测试 - 层次化门控注意力残差网络")
    print("   (双向注意力流 + 动态门控 + 层次残差)")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 70)
    
    # 获取全局call_llm（在OpenClaw会话中已经可用）
    import builtins
    if 'call_llm' not in builtins.__dict__:
        print("❌ 错误: call_llm 未定义，请在OpenClaw会话中运行")
        sys.exit(1)
    
    call_llm = builtins.__dict__['call_llm']
    
    start_time = time.time()
    
    # 🔥 初始化HGARN v1，启用所有创新
    agent = HGARMultiAgent(
        block_size=8,
        max_blocks_per_level=2,
        max_levels=3,
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        parallel_execution=True,
        max_parallel=4,
        enable_reverse_activation=True,
        enable_confidence_routing=True,
        min_gate_for_continue=0.15,
        llm_client=call_llm
    )
    
    print("\n🚀 开始运行 HGARN (层次化门控注意力残差)...\n")
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 70)
    print("📊 运行统计 HGARN:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 处理Levels: {len(result.hierarchical_levels)}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("🎯 最终结果 HGARN:\n")
    print(result.final_answer)
    print("=" * 70)
    
    # 打印各层次门控分数
    print("\n📈 各层次门控分数:")
    for lv in result.hierarchical_levels:
        print(f"  Level {lv.level_id}: gate = {lv.gate_score:.3f}")
    print()
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(__file__), f"test_result_hgarn_{int(time.time())}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# HGARN v1 层次化门控注意力残差网络 - 测试结果\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n")
        f.write(f"**问题:**\n\n{TEST_QUERY}\n\n")
        f.write("## 运行统计\n\n")
        f.write(f"- 处理Blocks: {result.blocks_processed}\n")
        f.write(f"- 处理Levels: {len(result.hierarchical_levels)}\n")
        f.write(f"- 分解子任务数: {result.subtasks_total}\n")
        f.write(f"- 估算token消耗: {result.total_tokens}\n")
        f.write(f"- 是否提前停止: {result.early_stopped}\n")
        f.write(f"- 运行时间: {elapsed:.2f} 秒\n\n")
        f.write("## 各层次门控分数\n\n")
        for lv in result.hierarchical_levels:
            f.write(f"- Level {lv.level_id}: **{lv.gate_score:.3f}**\n")
        f.write("\n## 最终结果\n\n")
        f.write(result.final_answer)
        f.write("\n")
    
    print(f"\n💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
