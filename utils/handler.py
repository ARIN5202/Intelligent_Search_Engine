import asyncio
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

DEFAULT_SYSTEM_PROMPT = """
你是一个专业的智能助手。请遵循以下要求：
1. 语言：根据用户的询问而定，用户是繁体中文提问，那么回答繁体中文，如果是英文提问，那么用英文。
2. 格式：不使用markdown，直接回答即可，不要携带任何符号 *，-这样的，纯文字即可。
3. 风格：回答应专业、客观、简洁明了。
4. 任务：你的主要任务是根据用户上传的文件内容和询问进行分析和回答。
"""

class AttachmentHandler:

    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-pro'):
        self.enabled = False
        if not HAS_GEMINI:
            print(" 功能不可用。")
            return

        if not api_key:
            print(" 功能不可用。")
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.enabled = True
            self.w = DEFAULT_SYSTEM_PROMPT
        except Exception as e:
            print(f" 配置失败: {e}")

    async def process(self, query: str, attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.enabled:
            return self._error_response("服务未启用或配置错误")

        try:
            content_parts = []

            # 1. 添加文本提示
            prompt_text = query if query else "请分析这些上传的文件内容。"
            content_parts.append(prompt_text)

            # 2. 处理并上传附件
            # 使用 asyncio.gather 并发上传文件可能会更快，但为了顺序稳定性，这里暂用循环
            for att in attachments:
                path_str = att.get('path')
                if not path_str:
                    continue

                file_path = Path(path_str)
                if not file_path.exists():
                    continue

                # 调用内部上传逻辑
                uploaded_file = await self._upload_file_async(file_path)
                if uploaded_file:
                    content_parts.append(uploaded_file)

            if len(content_parts) <= 1:  # 只有文本，没成功上传文件
                # 这种情况下也可以继续跑，或者报错，看你需求
                pass

            # 3. 调用生成 API
            response = await asyncio.to_thread(
                self.model.generate_content,
                content_parts
            )

            # 4. 返回结果
            return {
                'answer': response.text,
                'sources': [{'title': 'Gemini Flash Analysis', 'score': 1.0}],
                'confidence': 1.0,
                'preprocess': {'issues': []},  # 保持接口一致性
                'error': None
            }

        except Exception as e:
            return self._error_response(f"Gemini API 调用异常: {str(e)}")

    async def _upload_file_async(self, file_path: Path):
        """异步上传文件辅助函数"""
        try:
            # 自动推断 MIME
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/pdf' if file_path.suffix.lower() == '.pdf' else 'image/jpeg'

            # 运行同步的 upload_file 在线程池中
            uploaded_file = await asyncio.to_thread(
                genai.upload_file,
                path=file_path,
                mime_type=mime_type
            )

            # 等待处理完成
            while uploaded_file.state.name == "PROCESSING":
                await asyncio.sleep(1)
                uploaded_file = await asyncio.to_thread(genai.get_file, uploaded_file.name)

            return uploaded_file
        except Exception as e:
            print(f"⚠️ 文件上传失败 {file_path.name}: {e}")
            return None

    def _error_response(self, msg: str) -> Dict[str, Any]:
        return {
            'answer': f"处理失败: {msg}",
            'sources': [],
            'confidence': 0.0,
            'error': msg
        }