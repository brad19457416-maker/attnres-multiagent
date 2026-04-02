#!/usr/bin/env python3
"""
基于文件的客户端 - 利用当前 agent 会话逐步运行
每次需要 LLM 调用时，将 prompt 写入文件，等待 agent 读取回答并写入响应文件
"""

import os
import time
import json
from typing import Optional
import sys

# 工作目录
WORK_DIR = "/tmp/openclaw_llm_calls"
os.makedirs(WORK_DIR, exist_ok=True)


class FileBasedClient:
    """基于文件的客户端，适合分步测试"""
    
    def __init__(self, request_delay: float = 1.0):
        self.request_delay = request_delay
        self._counter = 0
        
    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """调用模型，通过文件传递"""
        
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
        
        # 写入请求文件
        request_file = os.path.join(WORK_DIR, f"request_{counter:03d}.txt")
        response_file = os.path.join(WORK_DIR, f"response_{counter:03d}.txt")
        
        with open(request_file, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        
        print(f"\n[FileBasedClient] Waiting for response (request #{counter}, file: {request_file})", file=sys.stderr)
        print(f"[FileBasedClient] Response should be written to: {response_file}", file=sys.stderr)
        sys.stderr.flush()
        
        # 等待响应文件出现
        max_wait = 3600  # 1小时足够
        waited = 0
        while not os.path.exists(response_file) or os.path.getsize(response_file) == 0:
            time.sleep(5)
            waited += 5
            if waited > max_wait:
                raise TimeoutError(f"Timeout waiting for response after {max_wait} seconds (request #{counter})")
        
        # 读取响应
        with open(response_file, "r", encoding="utf-8") as f:
            response = f.read().strip()
        
        # 清理（可选，保留用于调试）
        # try:
        #     os.unlink(request_file)
        #     os.unlink(response_file)
        # except:
        #     pass
        
        print(f"\n[FileBasedClient] Got response for request #{counter} ({len(response)} chars)", file=sys.stderr)
        sys.stderr.flush()
        
        return response


if __name__ == "__main__":
    # 测试
    client = FileBasedClient()
    print(f"Testing FileBasedClient...")
    result = client("你好，请介绍一下你自己")
    print(f"\nResponse:\n{result}")
