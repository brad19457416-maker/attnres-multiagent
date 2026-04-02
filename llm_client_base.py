"""
LLM Client 基类和接口定义
支持外部注入自定义 LLM 客户端（比如百度千帆、OpenAI 等）

用法:
    1. 继承 LLMClient 基类
    2. 实现 __call__ 方法
    3. 传入 HGARMultiAgent 构造函数

示例:
    class QianfanLLMClient(LLMClient):
        def __init__(self, api_key, secret_key):
            # 初始化千帆客户端
            ...
        
        def __call__(self, prompt: str, temperature: float = 0.7) -> str:
            # 调用千帆 API
            # 返回回答文本
            ...
    
    agent = HGARMultiAgent(
        llm_client=QianfanLLMClient(api_key, secret_key)
    )
"""

import json
import requests
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """LLM 客户端抽象基类
    
    所有自定义 LLM 客户端需要继承这个类并实现 __call__ 方法。
    """
    
    @abstractmethod
    def __call__(self, prompt: str, temperature: float = 0.7) -> str:
        """调用 LLM 生成回答
        
        Args:
            prompt: 输入prompt
            temperature: 采样温度
            
        Returns:
            生成的回答文本
        """
        pass


class QianfanCodingPlanClient(LLMClient):
    """百度千帆 Kimi Coding Plan 客户端
    
    OpenAI 兼容格式调用，正确路径: https://qianfan.baidubce.com/v2/chat/completions
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "kimi-k2.5",
    ):
        """
        Args:
            api_key: 完整的 API Key（bce-v3/... 格式）
            model: 模型名称，默认 kimi-k2.5
        """
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://qianfan.baidubce.com/v2/chat/completions"
    
    def __call__(self, prompt: str, temperature: float = 0.7) -> str:
        """调用千帆 Kimi API"""
        
        # BCE token 已经是 bce-v3/... 格式，不需要 Bearer 前缀
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=180,
        )
        
        response.raise_for_status()
        data = response.json()
        
        # OpenAI 兼容格式
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        elif "result" in data:
            return data["result"].strip()
        else:
            raise ValueError(f"Unexpected response format: {json.dumps(data, indent=2)}")
