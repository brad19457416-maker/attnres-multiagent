#!/usr/bin/env python3
"""
直接调用当前 OpenClaw agent 模型的客户端
通过临时文件和标准输入输出直接和当前 agent 通信
"""

import os
import time
import json
import sys
import tempfile
from typing import Optional

# 默认模型使用当前配置的 ark-code-latest
DEFAULT_MODEL = os.environ.get("OPENCLAW_MODEL", "ark/ark-code-latest")


class DirectClient:
    """直接在当前 OpenClaw 会话中调用模型
    
    这个客户端利用我们已经在 agent 会话中这个事实，直接通过
    将 prompt 写入临时文件然后让 agent 读取并回答。
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    """
    
    def __init__(self, model: Optional[str] = None, request_delay: float = 3.0):
        self.model = model or DEFAULT_MODEL
        self.request_delay = request_delay
        self._counter = 0
        
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用当前模型返回回答文本"""
        
        # 延迟避免请求过快
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        self._counter += 1
        counter = self._counter
        
        # 构建完整提示词
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # 将 prompt 写入临时文件
        request_file = f"/tmp/openclaw_direct_request_{counter}.md"
        with open(request_file, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        
        # 创建一个空响应文件来接收结果
        response_file = f"/tmp/openclaw_direct_response_{counter}.md"
        
        print(f"\n[DirectClient] Waiting for model response (request #{counter})...", file=sys.stderr)
        sys.stderr.flush()
        
        # 等待 agent 读取并写入响应
        # 最多等待 5 分钟
        for _ in range(60):
            time.sleep(5)
            if os.path.exists(response_file) and os.path.getsize(response_file) > 0:
                break
        
        if not os.path.exists(response_file) or os.path.getsize(response_file) == 0:
            raise TimeoutError(f"Timeout waiting for model response after 300 seconds (request #{counter})")
        
        # 读取响应
        with open(response_file, "r", encoding="utf-8") as f:
            response = f.read().strip()
        
        # 清理临时文件
        try:
            os.unlink(request_file)
            os.unlink(response_file)
        except:
            pass
        
        print(f"\n[DirectClient] Got response ({len(response)} chars)", file=sys.stderr)
        sys.stderr.flush()
        
        return response


if __name__ == "__main__":
    # 测试
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "你好，请介绍一下你自己"
    
    client = DirectClient()
    print(f"Calling model with prompt: {prompt}")
    print("\nWaiting for response...")
    # 在命令行测试需要手动写入响应文件
    result = client(prompt)
    print(f"\nResponse:\n{result}")
