#!/usr/bin/env python3
"""
测试 OpenClaw 模型调用
验证客户端是否能正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接使用当前模型，我们手动测试
from hierarchical_attn_res import HGARMultiAgent

# 我们手动在这里处理 LLM 调用
# 因为我们就在 agent 会话里，可以直接让模型回答每个子问题

def test_decomposition():
    """测试任务分解，我们手动调用模型"""
    
    from task_decomposer import TaskDecomposer
    
    TEST_QUERY = """请分析Python编程语言的主要优点，包括：
1. 语法特点
2. 应用领域
3. 生态系统
4. 对比其他语言的优势
"""
    
    print("=" * 70)
    print("🧪 测试 OpenClaw 模型调用 - 任务分解")
    print("=" * 70)
    print(f"\n问题:\n{TEST_QUERY}\n")
    
    # 我们需要得到分解提示词
    prompt = TaskDecomposer(None)._build_decompose_prompt(TEST_QUERY)
    print("\n📝 分解提示词:\n")
    print(prompt)
    print("\n" + "=" * 70)
    print("\n请模型回答这个提示词，我会将结果复制回来。")
    
    return prompt


if __name__ == "__main__":
    test_decomposition()
