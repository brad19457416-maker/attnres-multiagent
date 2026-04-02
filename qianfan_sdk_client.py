#!/usr/bin/env python3
"""
百度千帆 API 客户端 - 使用官方 SDK
适配 HGARN 的 llm_client 接口
"""

import os
import time
from typing import Optional
import qianfan

# 默认配置
DEFAULT_MODEL = "qianfan-code-latest"


class QianFanSDKClient:
    """百度千帆 官方 SDK 客户端，适配 HGARN llm_client 接口
    
    调用签名: llm_client(prompt, system_prompt=None, temperature=None) -> str
    
    需要设置环境变量:
    - QIANFAN_API_KEY: 你的百度千帆 API Key
    - QIANFAN_SECRET_KEY: (可选) 如果使用 AK/SK 认证
    - QIANFAN_MODEL: (可选) 模型名称，默认 qianfan-code-latest
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, 
                 model: Optional[str] = None, request_delay: float = 5.0):
        self.api_key = api_key or os.environ.get("QIANFAN_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("QIANFAN_SECRET_KEY", "")
        self.model = model or os.environ.get("QIANFAN_MODEL", DEFAULT_MODEL)
        self.request_delay = request_delay  # 请求间隔，避免限流
        
        if not self.api_key:
            raise ValueError("QIANFAN_API_KEY environment variable not set")
        
        # 初始化千帆客户端
        os.environ["QIANFAN_API_KEY"] = self.api_key
        if self.secret_key:
            os.environ["QIANFAN_SECRET_KEY"] = self.secret_key
        
        self.client = qianfan.ChatCompletion()
    
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用千帆 API 返回回答文本"""
        
        # 延迟避免限流
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        
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
        
        args = {
            "model": self.model,
            "messages": messages
        }
        
        if temperature is not None:
            args["temperature"] = temperature
        
        response = self.client.do(**args)
        
        # 千帆 SDK 返回格式
        return response.body["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    # 测试
    import sys
    client = QianFanSDKClient()
    print(f"Testing QianFanSDKClient with model {client.model}")
    result = client("你好，请介绍一下Python，一句话回答。")
    print(f"\nResponse:\n{result}")
