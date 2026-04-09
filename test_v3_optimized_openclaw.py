#!/usr/bin/env python3
"""
HGARN V3 (优化后) 单独测试 —— 使用同一个大问题验证优化效果
对比优化前 (V2) 和优化后 (V3) 的 token 消耗和运行时间
"""

import sys
import os
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试问题 - 和原来完全一样，保证公平对比
TEST_QUERY = """请帮我全面分析2025-2026年AI大模型领域的发展，包括:

1. 当前主要的技术路线分歧有哪些？比如开源闭源、MoE vs 密集模型、推理优化方向、架构创新等，每个路线详细分析
2. 对比各路线的优缺点，以及目前的性能/成本 trade-off
3. 国内外主要玩家（OpenAI、Anthropic、Google、字节、百度、阿里等）各自布局了哪些路线？
4. 商业化落地进展如何？有哪些成功的商业模式？
5. 预测未来12个月（到2027年3月）的发展趋势，有哪些技术会突破，哪些投资机会值得关注？

请全面深入分析，每个点都要有具体内容。
"""


def run_v3_test():
    """运行 HGARN V3 优化后测试"""
    from hierarchical_attn_res import HGARMultiAgent
    from openclaw_client import OpenClawClient
    
    print("\n" + "=" * 80)
    print("🔥 HGARN V3 (优化后) 测试 —— 层次化门控注意力残差网络 (OpenClaw 本地)")
    print("   包含 v3 第一阶段所有优化: JSON压缩+累积早停+合并小Block+WTA+容量限制+向量预过滤")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 80)
    
    print("\n🔧 初始化 OpenClaw 客户端...")
    client = OpenClawClient(request_delay=2.0)
    print("✓ 客户端初始化完成 (ark-code-latest)")
    
    start_time = time.time()
    
    # 🔥 HGARN V3 初始化，启用所有优化
    agent = HGARMultiAgent(
        block_size=8,
        max_blocks_per_level=2,
        max_levels=3,
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        parallel_execution=False,
        max_parallel=1,
        enable_reverse_activation=True,
        enable_confidence_routing=True,
        min_gate_for_continue=0.15,
        # v2 参数
        enable_working_memory_partition=True,
        gate_at_block_level=False,
        reverse_activation_gain_threshold=0.3,
        enable_dynamic_concurrency=True,
        min_concurrency=1,
        max_concurrency=8,
        # v3 新增优化参数 —— 全部默认开启
        cumulative_gain_threshold=2.5,
        enable_vector_prefilter=True,
        vector_similarity_threshold=0.3,
        llm_client=client
    )
    print("✓ HGARN V3 初始化完成，启用所有v3优化:")
    print(f"  - block_size = {agent.block_size}")
    print(f"  - max_levels = {agent.max_levels}")
    print(f"  - cumulative_gain_threshold = {agent.cumulative_gain_threshold}")
    print(f"  - enable_vector_prefilter = {agent.enable_vector_prefilter}")
    print(f"  - vector_similarity_threshold = {agent.vector_similarity_threshold}")
    
    print("\n🚀 开始运行 HGARN V3...\n")
    sys.stdout.flush()
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📊 运行统计 HGARN V3:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 处理Levels: {len(result.hierarchical_levels)}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 80)
    
    # 打印各层次门控分数
    print("\n📈 各层次门控分数:")
    for lv in result.hierarchical_levels:
        print(f"   Level {lv.level_id}: {lv.gate_score:.3f}")
    print()
    
    # 保存结果到文件
    output_file = os.path.join(
        os.path.dirname(__file__), 
        f"test_result_hgarnv3_openclaw_{int(time.time())}.md"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# HGARN V3 (优化后) 层次化门控注意力残差网络 - OpenClaw 本地模型测试结果\n\n")
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
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("🎯 最终结果 HGARN V3:\n")
    print(result.final_answer)
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("✅ HGARN V3 测试完成!")
    print(f"📄 结果文件: {output_file}")
    print("=" * 80)
    
    return {
        'result': result,
        'elapsed': elapsed,
        'output_file': output_file
    }


if __name__ == "__main__":
    run_v3_test()
