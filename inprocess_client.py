#!/usr/bin/env python3
"""
进程内客户端 - 直接利用当前运行的 agent 来调用模型
因为我们已经在 OpenClaw agent 会话中，可以直接通过全局回调调用模型
"""

import os
import time
import json
from typing import Optional
import sys

# 默认模型使用当前配置的 ark-code-latest
DEFAULT_MODEL = os.environ.get("OPENCLAW_MODEL", "ark/ark-code-latest")


# 全局存储，用于 agent 读取请求和写入响应
_current_request = None
_current_response = None


class InProcessClient:
    """进程内直接调用当前 agent 模型
    
    这个客户端利用我们已经在 agent 会话中这个事实，
    由 agent 直接生成回答，无需网络调用。
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    """
    
    def __init__(self, model: Optional[str] = None, request_delay: float = 1.0):
        self.model = model or DEFAULT_MODEL
        self.request_delay = request_delay
        self._counter = 0
        
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用当前模型返回回答文本"""
        
        global _current_request, _current_response
        
        self._counter += 1
        counter = self._counter
        
        # 延迟
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        # 构建完整提示词
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # 存储请求，等待当前 agent 读取并回答
        _current_request = full_prompt
        
        print(f"\n[InProcessClient] Waiting for agent response (#{counter})...", file=sys.stderr)
        sys.stderr.flush()
        
        # 在这里，我们触发一个异常让 agent 处理请求
        # agent 会设置 _current_response 然后重新运行
        if _current_response is None:
            # 通知 agent 我们需要一个回答
            raise NeedAgentResponse(f"Need agent response for request #{counter}")
        else:
            # 获取回答并清空
            response = _current_response
            _current_request = None
            _current_response = None
            return response


class NeedAgentResponse(Exception):
    """异常：需要 agent 提供回答"""
    pass


def get_current_request():
    """获取当前待回答的请求"""
    global _current_request
    return _current_request


def set_current_response(response: str):
    """设置回答结果"""
    global _current_response
    _current_response = response


if __name__ == "__main__":
    # 测试
    client = InProcessClient()
    print(f"Testing InProcessClient...")
    try:
        result = client("你好，请介绍一下你自己")
        print(f"Result: {result}")
    except NeedAgentResponse:
        print(f"\nAgent needs to provide response!")
        print(f"Request was:\n{get_current_request()}")
