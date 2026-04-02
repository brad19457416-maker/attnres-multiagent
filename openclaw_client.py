#!/usr/bin/env python3
"""
OpenClaw 配置的 Ark API 客户端 - 适配 HGARN 的 llm_client 接口
使用 OpenClaw 已配置的火山引擎 Ark API，绕过 Kimi API 限流
"""

import os
import time
import requests
import json
from typing import Optional

# 直接使用 OpenClaw 配置好的火山引擎 Ark 端点（OpenAI 兼容格式）
ARK_BASE = "https://ark.cn-beijing.volces.com/api/coding/v3"
ARK_API_KEY = "f0851eb9-c955-4052-a25e-f2eabf6a2330"

# 默认模型使用当前配置的 ark-code-latest
DEFAULT_MODEL = "ark/ark-code-latest"
# 火山引擎实际模型ID是 doubao-seed-code
DEFAULT_ARK_MODEL_ID = "doubao-seed-code"


class OpenClawClient:
    """OpenClaw 已配置的火山引擎 Ark API 客户端，适配 HGARN llm_client 接口
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    
    使用 OpenClaw 已配置的 API 密钥，直接调用火山引擎
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model: Optional[str] = None, request_delay: float = 3.0):
        self.api_key = api_key or ARK_API_KEY
        self.base_url = base_url or ARK_BASE
        # model 是 OpenClaw 内部名称，实际端点用 doubao-seed-code
        self.ark_model_id = "doubao-seed-code"
        self.request_delay = request_delay  # 请求间延迟，避免过快
        
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用火山引擎 Ark API 返回回答文本"""
        
        # 延迟避免请求过快
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
            "model": self.ark_model_id,
            "messages": messages
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        
        # 火山引擎 Ark 兼容OpenAI格式
        # 正确格式: https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions
        # 增加重试机制，网络不稳定时重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=600  # 增加到10分钟超时
                )
                response.raise_for_status()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"[WARN] Attempt {attempt+1} failed, retrying... ({e})")
                time.sleep(5 * (attempt + 1))
        
        response.raise_for_status()
        result = response.json()
        
        return result["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    # 测试
    client = OpenClawClient()
    result = client("你好，请介绍一下你自己")
    print(result)
