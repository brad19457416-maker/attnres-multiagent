#!/usr/bin/env python3
"""
测试百度千帆 Coding Plan API 连接
===
测试配置:
- MODEL: kimi-k2.5
- API KEY: bce-v3/ALTAKSP-...
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client_base import QianfanCodingPlanClient

# 填入你的配置
API_KEY = "bce-v3/ALTAKSP-F4CPMfAIahcHJ496r7adF/5db2956adbb4a629b306c3b063e11665233e0060"
MODEL = "moonshot-v1-8k"


def main():
    print("=== 测试百度千帆 Coding Plan 连接 ===")
    print(f"模型: {MODEL}")
    print(f"API KEY: {API_KEY[:20]}...{API_KEY[-20:]}")
    print()
    
    try:
        # 创建客户端
        client = QianfanCodingPlanClient(
            api_key=API_KEY,
            model=MODEL
        )
        print("✅ 客户端创建成功")
        print()
        
        # 简单测试请求
        prompt = "请用一句话回答：Python是什么？"
        print(f"发送测试请求: {prompt}")
        print()
        
        result = client(prompt, temperature=0.7)
        print("=== 响应 ===")
        print(result)
        print()
        print("✅ 测试成功！千帆 Coding Plan API 可以正常使用。")
        
    except Exception as e:
        print()
        print("❌ 测试失败！")
        print(f"错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
