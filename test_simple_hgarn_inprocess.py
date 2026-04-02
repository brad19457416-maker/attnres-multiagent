#!/usr/bin/env python3
"""
简单测试 HGARN - 使用进程内客户端直接调用当前 agent 模型
更小的问题，验证流程
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hierarchical_attn_res import HGARMultiAgent
from inprocess_client import InProcessClient, NeedAgentResponse, get_current_request, set_current_response

# 简单测试问题
TEST_QUERY = """请分析Python编程语言的主要优点，包括：
1. 语法特点
2. 应用领域
3. 生态系统
4. 对比其他语言的优势
"""


def main():
    print("=" * 70)
    print("🔥 HGARN v1 简单测试 (进程内直接调用当前 agent)")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 70)
    
    llm_client = InProcessClient()
    print("✓ InProcess 客户端初始化完成")
    print(f"  - 模型: {llm_client.model}")
    
    start_time = time.time()
    
    agent = HGARMultiAgent(
        block_size=4,
        max_blocks_per_level=2,
        max_levels=2,
        enable_recursive_decomposition=False,
        parallel_execution=False,
        enable_reverse_activation=True,
        enable_confidence_routing=True,
        min_gate_for_continue=0.15,
        llm_client=llm_client
    )
    print("✓ HGARN 初始化完成")
    
    print("\n🚀 开始运行...\n")
    sys.stdout.flush()
    
    try:
        result = agent.run(TEST_QUERY)
    except NeedAgentResponse:
        # 需要当前 agent 回答
        print("\n📡 NEED AGENT RESPONSE:")
        print("-" * 70)
        print(get_current_request())
        print("-" * 70)
        print("\n⚠️  请在上面对话中回答这个请求，然后继续执行。")
        return  # 退出，等待下次调用
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 70)
    print("📊 运行统计:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 处理Levels: {len(result.hierarchical_levels)}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("🎯 最终结果:\n")
    print(result.final_answer)
    print("=" * 70)
    
    print("\n📈 各层次门控分数:")
    for lv in result.hierarchical_levels:
        print(f"  Level {lv.level_id}: gate = {lv.gate_score:.3f}")
    
    output_file = os.path.join(os.path.dirname(__file__), f"test_simple_hgarn_inprocess_{int(time.time())}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# HGARN v1 简单测试结果 (InProcess)\n\n")
        f.write(f"**测试时间:** {datetime.now()}\n")
        f.write(f"**问题:**\n\n{TEST_QUERY}\n\n")
        f.write("## 运行统计\n\n")
        f.write(f"- 处理Blocks: {result.blocks_processed}\n")
        f.write(f"- 处理Levels: {len(result.hierarchical_levels)}\n")
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
