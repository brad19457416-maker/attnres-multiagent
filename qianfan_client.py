#!/usr/bin/env python3
"""
百度千帆 API 客户端 - 适配 HGARN 的 llm_client 接口
支持百度千帆 Coding Plan API Key
"""

import os
import time
import requests
import json
from typing import Optional

# 默认配置
DEFAULT_BASE_URL = "https://qianfan.baidubce.com/v2"
DEFAULT_MODEL = "kimi-k2.0"


class QianFanClient:
    """百度千帆 API 客户端，适配 HGARN llm_client 接口
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    
    需要设置环境变量:
    - QIANFAN_API_KEY: 你的百度千帆 API Key
    - QIANFAN_BASE_URL: (可选) 自定义端点
    - QIANFAN_MODEL: (可选) 模型名称，默认 qianfan-code-latest
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model: Optional[str] = None, request_delay: float = 3.0):
        self.api_key = api_key or os.environ.get("QIANFAN_API_KEY", "")
        self.base_url = base_url or os.environ.get("QIANFAN_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("QIANFAN_MODEL", DEFAULT_MODEL)
        self.request_delay = request_delay  # 请求间隔，避免限流
        
        if not self.api_key:
            raise ValueError("QIANFAN_API_KEY environment variable not set")
        
        # 确保 base_url 结尾没有斜杠
        self.base_url = self.base_url.rstrip('/')
        
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用千帆 API 返回回答文本"""
        
        # 延迟避免限流
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 构建 messages
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
        
        # 千帆 OpenAI 兼容端点
        url = f"{self.base_url}/chat/completions"
        
        data = {
            "model": self.model,
            "messages": messages
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=180
        )
        
        # 如果是 401，说明需要检查 API Key 权限
        if response.status_code == 401:
            error_msg = response.text
            if "coding_plan_api_key_not_allowed" in error_msg:
                raise PermissionError(
                    "此 API Key 为 Coding Plan 专用，仅允许调用 coding 端点，"
                    "不支持通用 chat/completions。请使用其他 API Key 或配置专用端点。"
                )
        
        response.raise_for_status()
        result = response.json()
        
        # 千帆返回格式兼容 OpenAI
        return result["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    # 测试
    import sys
    client = QianFanClient()
    print(f"Testing QianFanClient with model {client.model}")
    print(f"Base URL: {client.base_url}")
    result = client("你好，请介绍一下Python，一句话回答。")
    print(f"\nResponse:\n{result}")
