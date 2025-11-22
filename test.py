# test_azure.py
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://hkust.azure-api.net",
    api_key="59a2f6e35a454cd6aa3629992303d33a",
    api_version="2025-02-01-preview",
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",   # 或者你真实的 deployment 名，比如 "gpt-4o"
    messages=[
        {"role": "user", "content": "测试一下 Azure 连接是否正常"}
    ],
)

print(resp.choices[0].message.content)
