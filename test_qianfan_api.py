#!/usr/bin/env python3
"""
测试百度千帆 API 是否可用
"""

import os
import requests
import json

BASE_URL = "https://qianfan.baidubce.com/v2/chat/completions"
API_KEY = "bce-v3/ALTAKSP-R7a2xiHXVQ3W6OtqJpoPP/8e7c4dc422fc6d78e053da4026a130893391d83b"
MODEL = "kimi-k2.0"


def test_api():
    """测试千帆 API 连接"""
    print("=" * 60)
    print("🧪 测试百度千帆 API 连接")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"API Key length: {len(API_KEY)}")
    print()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    messages = [
        {
            "role": "user",
            "content": "你好，请用一句话介绍一下Python编程语言，不要超过50字。"
        }
    ]
    
    data = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("🚀 发送请求...")
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json=data,
            timeout=60
        )
        
        print(f"📡 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ 请求成功！")
            print(f"响应结构: {list(result.keys())}")
            
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                print(f"\n🤖 模型回复:\n{content}")
            else:
                print(f"\n📄 完整响应:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"\n❌ 请求失败")
            print(f"响应内容: {response.text}")
            
    except Exception as e:
        print(f"\n💥 异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_api()
