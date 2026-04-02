#!/usr/bin/env python3
"""
简单测试 v0 - 同样的问题，和 HGARN 对比
支持 Kimi API 或 OpenClaw 本地模型
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attnres_multiagent import AttnResMultiAgent

# 选择使用哪个客户端: "kimi" 或 "openclaw"
USE_OPENCLAW = os.environ.get("USE_OPENCLAW", "true").lower() == "true"

if USE_OPENCLAW:
    from openclaw_client import OpenClawClient
else:
    from kimi_client import KimiClient

# 同样的简单测试问题
TEST_QUERY = """请分析Python编程语言的主要优点，包括：
1. 语法特点
2. 应用领域
3. 生态系统
4. 对比其他语言的优势
"""


def main():
    print("=" * 70)
    print("attnres v0 简单测试 - 原始Block注意力残差")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 70)
    
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
    
    agent = AttnResMultiAgent(
        block_size=4,
        max_blocks=2,
        adaptive_early_stop=True,
        parallel_execution=False,
        enable_recursive_decomposition=False,
        max_recursion_depth=3,
        llm_client=llm_client
    )
    print("✓ v0 初始化完成")
    
    print("\n🚀 开始运行 v0...\n")
    sys.stdout.flush()
    
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 70)
    print("📊 运行统计 v0:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("🎯 最终结果 v0:\n")
    print(result.final_answer)
    print("=" * 70)
    
    output_file = os.path.join(os.path.dirname(__file__), f"test_simple_v0_{int(time.time())}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# attnres v0 简单测试结果\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n")
        f.write(f"**问题:**\n\n{TEST_QUERY}\n\n")
        f.write("## 运行统计\n\n")
        f.write(f"- 处理Blocks: {result.blocks_processed}\n")
        f.write(f"- 分解子任务数: {result.subtasks_total}\n")
        f.write(f"- 估算token消耗: {result.total_tokens}\n")
        f.write(f"- 是否提前停止: {result.early_stopped}\n")
        f.write(f"- 运行时间: {elapsed:.2f} 秒\n\n")
        f.write("## 结果\n\n")
        f.write(result.final_answer)
        f.write("\n")
    
    print(f"\n💾 结果已保存到: {output_file}")
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
