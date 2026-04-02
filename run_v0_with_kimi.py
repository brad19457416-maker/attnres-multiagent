#!/usr/bin/env python3
"""
attnres-multiagent v0 完整测试 - 使用 Kimi API
用于和 HGARN v1 对比
"""

import sys
import os
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attnres_multiagent import AttnResMultiAgent
from kimi_client import KimiClient

# 测试问题 - 和HGARN完全一样，保证对比公平
TEST_QUERY = """请帮我全面分析2025-2026年AI大模型领域的发展，包括:

1. 当前主要的技术路线分歧有哪些？比如开源闭源、MoE vs 密集模型、推理优化方向、架构创新等，每个路线详细分析
2. 对比各路线的优缺点，以及目前的性能/成本 trade-off
3. 国内外主要玩家（OpenAI、Anthropic、Google、字节、百度、阿里等）各自布局了哪些路线？
4. 商业化落地进展如何？有哪些成功的商业模式？
5. 预测未来12个月（到2027年3月）的发展趋势，有哪些技术会突破，哪些投资机会值得关注？

请全面深入分析，每个点都要有具体内容。"""


def main():
    print("=" * 80)
    print("attnres-multiagent v0 测试 - 原始Block注意力残差 (使用 Kimi API)")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 80)
    
    # 初始化 Kimi 客户端
    print("\n🔧 初始化 Kimi API 客户端...")
    kimi_client = KimiClient()
    print("✓ Kimi API 客户端初始化完成")
    
    start_time = time.time()
    
    # 初始化 v0
    agent = AttnResMultiAgent(
        block_size=8,
        max_blocks=3,
        adaptive_early_stop=True,
        parallel_execution=False,  # Kimi API不支持并行请求
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        llm_client=kimi_client
    )
    print("✓ v0 初始化完成，参数:")
    print(f"  - block_size = {agent.block_size}")
    print(f"  - max_blocks = {agent.max_blocks}")
    print(f"  - adaptive_early_stop = {agent.adaptive_early_stop}")
    print(f"  - recursive_decomposition = {agent.enable_recursive_decomposition}")
    
    print("\n🚀 开始运行 v0 (原始Block注意力残差)...\n")
    sys.stdout.flush()
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📊 运行统计 v0:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("🎯 最终结果 v0:\n")
    print(result.final_answer)
    print("=" * 80)
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(__file__), f"test_result_v0_kimi_{int(time.time())}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# attnres-multiagent v0 - Kimi API 测试结果\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n")
        f.write(f"**问题:**\n\n{TEST_QUERY}\n\n")
        f.write("## 运行统计\n\n")
        f.write(f"- 处理Blocks: {result.blocks_processed}\n")
        f.write(f"- 分解子任务数: {result.subtasks_total}\n")
        f.write(f"- 估算token消耗: {result.total_tokens}\n")
        f.write(f"- 是否提前停止: {result.early_stopped}\n")
        f.write(f"- 运行时间: {elapsed:.2f} 秒\n\n")
        f.write("## 最终结果\n\n")
        f.write(result.final_answer)
        f.write("\n")
    
    print(f"\n💾 结果已保存到: {output_file}")
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
