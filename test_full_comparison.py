#!/usr/bin/env python3
"""
完整对比测试: attnres v0 vs HGARN v1
测试问题: 分析2025-2026年AI大模型领域的主要技术创新路线图
这个脚本会依次运行两个版本，输出完整对比结果
支持 Kimi API 或 OpenClaw 本地模型
"""

import sys
import os
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 选择使用哪个客户端: "kimi" 或 "openclaw"
USE_OPENCLAW = os.environ.get("USE_OPENCLAW", "true").lower() == "true"

# 测试问题 - 大而复杂的问题，真正体现两种架构的差异
TEST_QUERY = """请帮我全面分析2025-2026年AI大模型领域的发展，包括:

1. 当前主要的技术路线分歧有哪些？比如开源闭源、MoE vs 密集模型、推理优化方向、架构创新等，每个路线详细分析
2. 对比各路线的优缺点，以及目前的性能/成本 trade-off
3. 国内外主要玩家（OpenAI、Anthropic、Google、字节、百度、阿里等）各自布局了哪些路线？
4. 商业化落地进展如何？有哪些成功的商业模式？
5. 预测未来12个月（到2027年3月）的发展趋势，有哪些技术会突破，哪些投资机会值得关注？

请全面深入分析，每个点都要有具体内容。"""


def run_v0():
    """运行 attnres v0"""
    from attnres_multiagent import AttnResMultiAgent
    
    if USE_OPENCLAW:
        from openclaw_client import OpenClawClient
    else:
        from kimi_client import KimiClient
    
    print("\n" + "=" * 80)
    print("🟢 attnres-multiagent v0 - 原始Block注意力残差")
    print("=" * 80)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 80)
    
    start_time = time.time()
    
    if USE_OPENCLAW:
        llm_client = OpenClawClient()
        print(f"✓ OpenClaw 本地模型客户端初始化完成")
        print(f"  - 模型: {llm_client.model}")
    else:
        import os
        api_key = os.environ.get("KIMI_API_KEY")
        print(f"\n🔧 API Key length: {len(api_key) if api_key else 'NOT FOUND'}")
        llm_client = KimiClient()
        print("✓ Kimi API 客户端初始化完成")
    
    # 初始化agent v0
    agent = AttnResMultiAgent(
        block_size=8,
        max_blocks=3,
        adaptive_early_stop=True,
        parallel_execution=False,  # OpenClaw 本地模型不支持并行
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        llm_client=llm_client
    )
    
    print("\n🚀 开始运行 v0...\n")
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    stats = {
        "blocks_processed": result.blocks_processed,
        "subtasks_total": result.subtasks_total,
        "total_tokens": result.total_tokens,
        "early_stopped": result.early_stopped,
        "elapsed_seconds": elapsed,
        "final_answer": result.final_answer
    }
    
    print("\n" + "=" * 80)
    print("📊 v0 运行统计:")
    print(f"  - 处理Blocks: {stats['blocks_processed']}")
    print(f"  - 分解子任务数: {stats['subtasks_total']}")
    print(f"  - 估算token消耗: {stats['total_tokens']}")
    print(f"  - 是否提前停止: {stats['early_stopped']}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 80)
    
    # 保存结果
    output_file = f"test_full_v0_{int(time.time())}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# attnres-multiagent v0 测试结果\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n")
        f.write(f"**问题:**\n\n{TEST_QUERY}\n\n")
        f.write("## 运行统计\n\n")
        f.write(f"- 处理Blocks: {stats['blocks_processed']}\n")
        f.write(f"- 分解子任务数: {stats['subtasks_total']}\n")
        f.write(f"- 估算token消耗: {stats['total_tokens']}\n")
        f.write(f"- 是否提前停止: {stats['early_stopped']}\n")
        f.write(f"- 运行时间: {elapsed:.2f} 秒\n\n")
        f.write("## 最终结果\n\n")
        f.write(stats['final_answer'])
        f.write("\n")
    
    print(f"\n💾 v0 结果已保存到: {output_file}")
    
    return stats


def run_hgarn():
    """运行 HGARN v1"""
    from hierarchical_attn_res import HGARMultiAgent
    
    print("\n" + "=" * 80)
    print("🔴 HGARN v1 - 层次化门控注意力残差网络")
    print("=" * 80)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 80)
    
    start_time = time.time()
    
    if USE_OPENCLAW:
        from openclaw_client import OpenClawClient
        llm_client = OpenClawClient()
        print(f"✓ OpenClaw 本地模型客户端初始化完成")
        print(f"  - 模型: {llm_client.model}")
    else:
        from kimi_client import KimiClient
        import os
        api_key = os.environ.get("KIMI_API_KEY")
        print(f"\n🔧 API Key length: {len(api_key) if api_key else 'NOT FOUND'}")
        llm_client = KimiClient()
        print("✓ Kimi API 客户端初始化完成")
    
    # 初始化agent HGARN
    agent = HGARMultiAgent(
        block_size=8,
        max_blocks_per_level=2,
        max_levels=3,
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        parallel_execution=False,  # OpenClaw 本地模型不支持并行
        enable_reverse_activation=True,
        enable_confidence_routing=True,
        min_gate_for_continue=0.15,
        llm_client=llm_client
    )
    
    print("\n🚀 开始运行 HGARN v1...\n")
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    stats = {
        "blocks_processed": result.blocks_processed,
        "levels_processed": len(result.hierarchical_levels) if hasattr(result, 'hierarchical_levels') else 0,
        "subtasks_total": result.subtasks_total,
        "total_tokens": result.total_tokens,
        "early_stopped": result.early_stopped if hasattr(result, 'early_stopped') else False,
        "elapsed_seconds": elapsed,
        "final_answer": result.final_answer
    }
    
    print("\n" + "=" * 80)
    print("📊 HGARN v1 运行统计:")
    print(f"  - 处理Blocks: {stats['blocks_processed']}")
    print(f"  - 处理Levels: {stats['levels_processed']}")
    print(f"  - 分解子任务数: {stats['subtasks_total']}")
    print(f"  - 估算token消耗: {stats['total_tokens']}")
    print(f"  - 是否提前停止: {stats['early_stopped']}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 80)
    
    # 保存结果
    output_file = f"test_full_hgarn_{int(time.time())}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# HGARN v1 测试结果\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n")
        f.write(f"**问题:**\n\n{TEST_QUERY}\n\n")
        f.write("## 运行统计\n\n")
        f.write(f"- 处理Blocks: {stats['blocks_processed']}\n")
        f.write(f"- 处理Levels: {stats['levels_processed']}\n")
        f.write(f"- 分解子任务数: {stats['subtasks_total']}\n")
        f.write(f"- 估算token消耗: {stats['total_tokens']}\n")
        f.write(f"- 是否提前停止: {stats['early_stopped']}\n")
        f.write(f"- 运行时间: {elapsed:.2f} 秒\n\n")
        f.write("## 最终结果\n\n")
        f.write(stats['final_answer'])
        f.write("\n")
    
    print(f"\n💾 HGARN 结果已保存到: {output_file}")
    
    return stats


def main():
    """主函数 - 依次运行两个版本"""
    print("\n" + "=" * 80)
    print("🔥 完整对比测试: attnres v0 vs HGARN v1")
    print("=" * 80)
    print(f"\n测试问题: 分析2025-2026年AI大模型领域发展")
    print(f"\n将依次运行 v0 和 HGARN v1，生成完整对比数据\n")
    
    # 先运行 v0
    v0_stats = run_v0()
    
    # 等待一下，避免API限流
    print("\n⏳ 等待 10 秒后运行HGARN...")
    time.sleep(10)
    
    # 再运行 HGARN
    hgarn_stats = run_hgarn()
    
    # 输出最终对比
    print("\n" + "=" * 80)
    print("🏁 测试完成! 最终对比:")
    print("=" * 80)
    print(f"\n{'指标':<20} {'v0':<15} {'HGARN v1':<15}")
    print("-" * 50)
    print(f"{'Blocks':<20} {v0_stats['blocks_processed']:<15} {hgarn_stats['blocks_processed']:<15}")
    print(f"{'Levels':<20} {'-':<15} {hgarn_stats['levels_processed']:<15}")
    print(f"{'子任务数':<20} {v0_stats['subtasks_total']:<15} {hgarn_stats['subtasks_total']:<15}")
    print(f"{'Token消耗':<20} {v0_stats['total_tokens']:<15} {hgarn_stats['total_tokens']:<15}")
    print(f"{'运行时间(秒)':<20} {v0_stats['elapsed_seconds']:.2f} {hgarn_stats['elapsed_seconds']:.2f}")
    print(f"{'提前停止':<20} {str(v0_stats['early_stopped']):<15} {str(hgarn_stats['early_stopped']):<15}")
    print("-" * 50)
    
    # 计算差异
    token_diff = (hgarn_stats['total_tokens'] - v0_stats['total_tokens']) / v0_stats['total_tokens'] * 100
    time_diff = (hgarn_stats['elapsed_seconds'] - v0_stats['elapsed_seconds']) / v0_stats['elapsed_seconds'] * 100
    
    print(f"\n📈 相对差异:")
    print(f"  - Token: {token_diff:+.1f}%")
    print(f"  - 时间: {time_diff:+.1f}%")
    print("=" * 80)
    
    # 保存对比总结
    summary_file = f"comparison_summary_{int(time.time())}.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# 完整对比测试总结: attnres v0 vs HGARN v1\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n\n")
        f.write("## 测试问题\n\n")
        f.write(TEST_QUERY)
        f.write("\n\n## 对比结果\n\n")
        f.write("| 指标 | attnres v0 | HGARN v1 |\n")
        f.write("|------|------------|----------|\n")
        f.write(f"| Blocks | {v0_stats['blocks_processed']} | {hgarn_stats['blocks_processed']} |\n")
        f.write(f"| Levels | - | {hgarn_stats['levels_processed']} |\n")
        f.write(f"| 子任务数 | {v0_stats['subtasks_total']} | {hgarn_stats['subtasks_total']} |\n")
        f.write(f"| Token消耗 | {v0_stats['total_tokens']} | {hgarn_stats['total_tokens']} |\n")
        f.write(f"| 运行时间(秒) | {v0_stats['elapsed_seconds']:.2f} | {hgarn_stats['elapsed_seconds']:.2f} |\n")
        f.write(f"| 提前停止 | {v0_stats['early_stopped']} | {hgarn_stats['early_stopped']} |\n\n")
        f.write("## 相对差异\n\n")
        f.write(f"- Token: **{token_diff:+.1f}%**\n")
        f.write(f"- 时间: **{time_diff:+.1f}%**\n")
    
    print(f"\n📝 对比总结已保存到: {summary_file}")
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
