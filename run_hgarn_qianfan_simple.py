#!/usr/bin/env python3
"""
HGARN v1 完整测试 - 使用百度千帆 Kimi API
直接使用 requests，适配千帆 v2 API + Coding Plan API Key
"""

import sys
import os
import time
import requests
import json
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hierarchical_attn_res import HGARMultiAgent

# ============ 配置 ============
# 这是你提供的千帆 Coding Plan API Key
QIANFAN_API_KEY = "bce-v3/ALTAKSP-R7a2xiHXVQ3W6OtqJpoPP/8e7c4dc422fc6d78e053da4026a130893391d83b"
# 根据千帆文档，v2 API 的端点是 https://qianfan.baidubce.com/v2
# 对于 Coding Plan，模型是 qianfan-code-latest
QIANFAN_BASE_URL = "https://qianfan.baidubce.com/v2"
QIANFAN_MODEL = "qianfan-code-latest"
REQUEST_DELAY = 8.0  # 间隔8秒，避免限流


class QianFanV2Client:
    """百度千帆 v2 API 客户端，适配 HGARN llm_client 接口"""
    
    def __init__(self):
        self.api_key = QIANFAN_API_KEY
        self.base_url = QIANFAN_BASE_URL
        self.model = QIANFAN_MODEL
        self.request_delay = REQUEST_DELAY
    
    def __call__(self, prompt: str, system_prompt: str = None, temperature: float = None) -> str:
        """调用千帆 v2 API 返回回答文本"""
        
        # 延迟避免限流
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        data = {
            "model": self.model,
            "messages": messages
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        
        url = f"{self.base_url}/chat/completions"
        print(f"  [QianFan] 调用 {self.model} ...", file=sys.stderr)
        
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=300
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()


# 测试问题 - 用户提供
TEST_QUERY = """我最近了解到KIMI发布的关于注意力残差的概念，以及字节的DEERFLOW 2.0的架构，我觉得这些AI方面的前沿理论非常有研究价值。请帮我整理近1年以来，前沿的AI理论与实践应用案例，并分析其中的优劣势，以及结合当前整体AI产业链（包含上下游）的情况，对未来1年的AI发展方向提出合理猜想。"""


def main():
    print("=" * 80)
    print("🔥 HGARN v1 测试 - 层次化门控注意力残差网络 (百度千帆 API)")
    print("   双向注意力流 + 动态门控 + 层次残差")
    print("测试时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    print(f"\n测试问题:\n{TEST_QUERY}\n")
    print("-" * 80)
    
    # 初始化客户端
    print("\n🔧 初始化百度千帆 API 客户端...")
    client = QianFanV2Client()
    print(f"✓ 模型: {QIANFAN_MODEL}")
    print("✓ 客户端初始化完成")
    
    start_time = time.time()
    
    # 🔥 初始化 HGARN v1，启用所有创新
    agent = HGARMultiAgent(
        block_size=8,
        max_blocks_per_level=2,
        max_levels=3,
        enable_recursive_decomposition=True,
        max_recursion_depth=3,
        parallel_execution=False,  # API 限流，串行执行
        max_parallel=1,
        enable_reverse_activation=True,
        enable_confidence_routing=True,
        min_gate_for_continue=0.15,
        llm_client=client
    )
    print("✓ HGARN 初始化完成，参数:")
    print(f"  - block_size = {agent.block_size}")
    print(f"  - max_levels = {agent.max_levels}")
    print(f"  - reverse_activation = {agent.enable_reverse_activation}")
    print(f"  - confidence_routing = {agent.enable_confidence_routing}")
    
    print("\n🚀 开始运行 HGARN (层次化门控注意力残差)...\n")
    sys.stdout.flush()
    
    # 运行
    result = agent.run(TEST_QUERY)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📊 运行统计 HGARN:")
    print(f"  - 处理Blocks: {result.blocks_processed}")
    print(f"  - 处理Levels: {len(result.hierarchical_levels)}")
    print(f"  - 分解子任务数: {result.subtasks_total}")
    print(f"  - 估算token消耗: {result.total_tokens}")
    print(f"  - 是否提前停止: {result.early_stopped}")
    print(f"  - 运行时间: {elapsed:.2f} 秒")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("🎯 最终结果 HGARN:\n")
    print(result.final_answer)
    print("=" * 80)
    
    # 打印各层次门控分数
    print("\n📈 各层次门控分数:")
    for lv in result.hierarchical_levels:
        print(f"  Level {lv.level_id}: gate = {lv.gate_score:.3f}")
    print()
    
    # 保存结果到文件
    output_file = os.path.join(os.path.dirname(__file__), f"test_result_hgarn_qianfan_{int(time.time())}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# HGARN v1 层次化门控注意力残差网络 - 百度千帆 API 测试结果\n\n")
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
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
