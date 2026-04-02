#!/usr/bin/env python3
"""
Kimi API 客户端 - 适配 HGARN 的 llm_client 接口
"""

import os
import time
import requests
import json
from typing import Optional

KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
KIMI_BASE_URL = "https://api.moonshot.cn/v1"


class KimiClient:
    """Kimi API 客户端，适配 HGARN llm_client 接口
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, request_delay: float = 10.0):
        self.api_key = api_key or KIMI_API_KEY
        self.base_url = base_url or KIMI_BASE_URL
        self.request_delay = request_delay  # 请求间延迟，避免限流
        
        if not self.api_key:
            raise ValueError("KIMI_API_KEY environment variable not set")
    
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用 Kimi API 返回回答文本"""
        
        # 延迟避免限流
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
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
            "model": "moonshot-v1-8k",
            "messages": messages
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=180
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    # 测试
    client = KimiClient()
    result = client("你好，请介绍一下你自己")
    print(result)
