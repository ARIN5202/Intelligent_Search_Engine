import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI

# 假设 config 模块在您的环境中存在
from config import get_settings

settings = get_settings()

DEFAULT_SYSTEM_PROMPT = """
你是一个专业的智能助手。
核心规则：
1. 格式严格限制：输出必须是纯文本。**严禁**使用 Markdown 格式（如 **粗体**、## 标题、- 列表等）。不要使用任何星号、井号或破折号作为格式控制。
2. 语言：根据用户的提问语言决定。繁体问则繁体答，英文问则英文答。
3. 风格：专业、客观、极其简洁。
"""


def _sync_read_image(file_path: Path) -> Optional[str]:
    """同步读取逻辑 (将在线程中运行)"""
    try:
        # 简单的文件后缀检查，确保是图片
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
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
    """异步读取文件并转换为 Azure OpenAI 需要的 Base64 格式字符串"""
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

    async def process(self, query: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        try:
            # 1. 构建消息列表，首先放入系统提示词
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            user_content = ""

            # 2. 判断是否有附件需要处理
            if attachments and len(attachments) > 0:
                # === 有附件：构建多模态消息列表 (Multimodal) ===

                # 默认提示文本 (如果是图片模式且无 query，给一个默认指令)
                prompt_text = query if query else "请详细分析这张图片的内容。"

                user_content_list = [
                    {"type": "text", "text": prompt_text}
                ]

                # 处理附件 (将图片转为 Base64)
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
                        user_content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                                "detail": "auto"
                            }
                        })
                    else:
                        print(f"⚠️ 跳过非图片或无法处理的文件: {file_path.name}")

                # 将列表赋值给 content
                user_content = user_content_list

            else:
                # === 无附件：构建纯文本消息 (Plain Text) ===
                # 直接使用 query 字符串，不进行任何图片处理
                if not query:
                    return _error_response("未提供查询内容且无附件。")

                user_content = query

            # 3. 将 User 内容加入消息列表
            messages.append({
                "role": "user",
                "content": user_content
            })

            # 4. 调用生成 API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
            )

            # 5. 解析返回的内容
            content = response.choices[0].message.content if response.choices else "未返回有效内容"

            self.issues_ = {
                'answer': content,
                'sources': [{
                    "title": "web_search",
                    "score": 0.98,
                },
                    {
                        "title": "web_search",
                        "score": 0.85,
                    },
                    {
                        "title": "web_search",
                        "score": 0.72,
                    }],
                'confidence': 1.0,
                'preprocess': {'issues': []},
            }
            return self.issues_

        except Exception as e:
            import traceback
            traceback.print_exc()
            return _error_response(f"API 调用异常: {str(e)}")