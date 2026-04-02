#!/usr/bin/env python3
"""
只运行 v0 - 分开测试避免限流
"""

import sys
import os
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置API Key - 必须在导入之前
os.environ["KIMI_API_KEY"] = "sk-SscLqVqagJkcfEjgbRVjDn3MhRNRLYdGVuiRv79pFjlAc2c5"

from attnres_multiagent import AttnResMultiAgent
from kimi_client import KimiClient

# 测试问题 - 大而复杂的问题
TEST_QUERY = """请帮我全面分析2025-2026年AI大模型领域的发展，包括:

1. 当前主要的技术路线分歧有哪些？比如开源闭源、MoE vs 密集模型、推理优化方向、架构创新等，每个路线详细分析
2. 对比各路线的优缺点，以及目前的性能/成本 trade-off
3. 国内外主要玩家（OpenAI、Anthropic、Google、字节、百度、阿里等）各自布局了哪些路线？
4. 商业化落地进展如何？有哪些成功的商业模式？
5. 预测未来12个月（到2027年3月）的发展趋势，有哪些技术会突破，哪些投资机会值得关注？

请全面深入分析，每个点都要有具体内容。"""


def main():
    """主函数 - 只运行 v0"""
    print("\n" + "=" * 80)
    print("🟢 只运行 attnres v0 - 原始Block注意力残差")
    print("=" * 80)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    
    # 设置API Key
    os.environ["KIMI_API_KEY"] = "sk-SscLqVqagJkcfEjgbRVjDn3MhRNRLYdGVuiRv79pFjlAc2c5"
    
    api_key = os.environ.get("KIMI_API_KEY")
    print(f"\n🔧 API Key length: {len(api_key) if api_key else 'NOT FOUND'}")
    
    kimi_client = KimiClient()
    print("✓ Kimi API 客户端初始化完成")
    
    start_time = time.time()
    
    # 初始化agent v0 - 关闭并行，降低请求频率避免限流
    agent = AttnResMultiAgent(
        block_size=8,
        max_blocks=3,
        adaptive_early_stop=True,
        parallel_execution=False,  # 关闭并行， sequential 执行，降低限流概率
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        llm_client=kimi_client
    )
    print("✓ v0 初始化完成")
    
    print("\n🚀 开始运行 v0...\n")
    sys.stdout.flush()
    
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
    
    print("\n" + "=" * 80)
    print("🎯 最终结果 v0:\n")
    print(stats['final_answer'])
    print("=" * 80)
    
    # 保存结果到文件
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
    print("\n✅ v0 测试完成!")


if __name__ == "__main__":
    main()
