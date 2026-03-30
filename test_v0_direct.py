#!/usr/bin/env python3
"""
直接运行v0测试，不依赖模块安装
"""

import sys
import os
import time
from datetime import datetime

# 直接导入，不需要相对导入
exec(open(os.path.join(os.path.dirname(__file__), "attn_types.py")).read())
exec(open(os.path.join(os.path.dirname(__file__), "task_decomposer.py")).read())
exec(open(os.path.join(os.path.dirname(__file__), "subagent_executor.py")).read())
exec(open(os.path.join(os.path.dirname(__file__), "attention_aggregator.py")).read())
exec(open(os.path.join(os.path.dirname(__file__), "attnres_multiagent.py")).read())

# 测试问题
TEST_QUERY = """请帮我全面分析2025-2026年AI大模型领域的发展，包括:

1. 当前主要的技术路线分歧有哪些？比如开源闭源、MoE vs 密集模型、推理优化方向、架构创新等，每个路线详细分析
2. 对比各路线的优缺点，以及目前的性能/成本 trade-off
3. 国内外主要玩家（OpenAI、Anthropic、Google、字节、百度、阿里等）各自布局了哪些路线？
4. 商业化落地进展如何？有哪些成功的商业模式？
5. 预测未来12个月（到2027年3月）的发展趋势，有哪些技术会突破，哪些投资机会值得关注？

请全面深入分析，每个点都要有具体内容。"""


def main():
    print("=" * 70)
    print("attnres-multiagent v0 测试 - 原始Block注意力残差")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 70)
    
    start_time = time.time()
    
    # 初始化agent v0
    agent = AttnResMultiAgent(
        block_size=8,
        max_blocks=3,
        adaptive_early_stop=True,
        parallel_execution=True,
        enable_recursive_decomposition=True,
        max_recursion_depth=3
    )
    
    print("\n🚀 开始运行...\n")
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 70)
    print("📊 运行统计:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("🎯 最终结果:\n")
    print(result.final_answer)
    print("=" * 70)
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(__file__), f"test_result_v0_{int(time.time())}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# attnres-multiagent v0 测试结果\n\n")
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


if __name__ == "__main__":
    main()
