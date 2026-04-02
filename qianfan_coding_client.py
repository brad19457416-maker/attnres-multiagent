#!/usr/bin/env python3
"""
百度千帆 Coding Plan API 客户端 - 适配 HGARN 的 llm_client 接口
Coding Plan 使用专用端点
"""

import os
import time
import requests
import json
from typing import Optional

# 默认配置
DEFAULT_BASE_URL = "https://qianfan.baidubce.com/v2/coding"
DEFAULT_MODEL = "kimi-k2.0"


class QianFanCodingClient:
    """百度千帆 Coding Plan API 客户端，适配 HGARN llm_client 接口
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    
    需要设置环境变量:
    - QIANFAN_API_KEY: 你的百度千帆 Coding Plan API Key
    - QIANFAN_BASE_URL: (可选) 自定义端点
    - QIANFAN_MODEL: (可选) 模型名称，默认 kimi-k2.0
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model: Optional[str] = None, request_delay: float = 5.0):
        self.api_key = api_key or os.environ.get("QIANFAN_API_KEY", "")
        self.base_url = base_url or os.environ.get("QIANFAN_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("QIANFAN_MODEL", DEFAULT_MODEL)
        self.request_delay = request_delay  # 请求间隔，避免限流
        
        if not self.api_key:
            raise ValueError("QIANFAN_API_KEY environment variable not set")
        
        # 确保 base_url 结尾没有斜杠
        self.base_url = self.base_url.rstrip('/')
        
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用千帆 Coding API 返回回答文本"""
        
        # 延迟避免限流
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Coding Plan 格式：messages 数组
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
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=data,
            timeout=300
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Coding 返回格式兼容 OpenAI
        return result["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    # 测试
    import sys
    client = QianFanCodingClient()
    print(f"Testing QianFanCodingClient with model {client.model}")
    print(f"Base URL: {client.base_url}")
    result = client("你好，请介绍一下Python，一句话回答。")
    print(f"\nResponse:\n{result}")
