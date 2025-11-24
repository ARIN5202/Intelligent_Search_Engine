import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

from config import get_settings

settings = get_settings()

DEFAULT_SYSTEM_PROMPT = """
你是一个专业的智能助手。
核心规则：
1. 格式严格限制：输出必须是纯文本。**严禁**使用 Markdown 格式（如 **粗体**、## 标题、- 列表等）。不要使用任何星号、井号或破折号作为格式控制。
2. 语言：根据用户的提问语言决定。繁体问则繁体答，英文问则英文答。
3. 引用要求：请在回答中使用 [1], [2] 这样的形式在句尾标注信息来源。
4. 风格：专业、客观、极其简洁。
"""


def _sync_read_image(file_path: Path) -> Optional[str]:
    """同步读取逻辑 (将在线程中运行)"""
    try:
        # 简单的文件后缀检查，确保是图片
        # 注意：Azure 支持 jpeg, jpg, png, gif, webp
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            # 如果需要支持文本文件的内容读取，可以在这里扩展逻辑
            return None

        # 猜测 MIME 类型
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "image/jpeg"  # 默认兜底

        with open(file_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{base64_encoded}"

    except Exception as e:
        print(f"⚠️ 图片读取失败 {file_path.name}: {e}")
        return None


async def _read_image_as_base64(file_path: Path) -> Optional[str]:
    """
    异步读取文件并转换为 Azure OpenAI 需要的 Base64 格式字符串
    返回格式: data:image/jpeg;base64,......
    """
    return await asyncio.to_thread(_sync_read_image, file_path)


def _error_response(msg: str) -> Dict[str, Any]:
    return {
        'answer': f"处理失败: {msg}",
        'sources': [],
        'confidence': 0.0,
        'error': msg
    }


class AttachmentHandler:
    def __init__(self, model_name: str = 'gpt-4o',
                 system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT):

        self.model_name = model_name
        self.system_prompt = system_prompt

        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_url,
            api_key=settings.azure_api_key,
            api_version="2025-02-01-preview",
        )

    async def process(self, query: str, attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # 1. 构建消息列表，首先放入系统提示词
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # 2. 构建 User 消息的内容列表 (这是一个包含文本和图片的混合列表)
            # 默认提示文本
            prompt_text = query if query else "请详细分析这张图片的内容。"
            user_content_list = [
                {"type": "text", "text": prompt_text}
            ]

            # 3. 处理附件 (将图片转为 Base64)
            for att in attachments:
                path_str = att.get('path')
                if not path_str: continue

                file_path = Path(path_str)
                if not file_path.exists():
                    print(f"⚠️ 文件 {file_path} 不存在。")
                    continue

                # 异步读取并转换图片
                image_data = await _read_image_as_base64(file_path)

                if image_data:
                    # 添加图片到消息体中
                    user_content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_data,  # 格式如: data:image/jpeg;base64,xxxx
                            "detail": "auto"  # 让模型自动决定分辨率模式
                        }
                    })
                else:
                    # 如果不是图片或处理失败，可以选择将其作为纯文本读取（可选）
                    print(f"⚠️ 跳过非图片或无法处理的文件: {file_path.name}")

            # 4. 将构建好的 User 内容加入消息列表
            messages.append({
                "role": "user",
                "content": user_content_list
            })

            # 5. 调用生成 API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
            )

            # 6. 解析返回的内容
            self.issues_ = {
                'answer': response.choices[0].message.content if response.choices else "未返回有效内容",
                'sources': [{
                    "title": "web_search",
                    "score": 0.98,  # 高相关度
                },
                    {
                        "title": "web_search",
                        "score": 0.85,  # 中等相关度
                    },
                    {
                        "title": "web_search",
                        "score": 0.72,  # 较低相关度
                    }],
                'confidence': 1.0,
                'preprocess': {'issues': []},
            }
            return self.issues_

        except Exception as e:
            import traceback
            traceback.print_exc()
            return _error_response(f"API 调用异常: {str(e)}")

