#!/usr/bin/env python3
"""测试 OpenClaw 本地模型连接"""

from openclaw_client import OpenClawClient

print("🔧 测试 OpenClaw 客户端连接...")
client = OpenClawClient(request_delay=0)

print("\n🚀 发送测试请求...")
result = client("你好，请用一句话介绍你自己")
print(f"\n✅ 连接成功！返回结果:\n{result}")
print("\n📊 连接测试通过！")
