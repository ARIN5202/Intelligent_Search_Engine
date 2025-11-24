#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能代理框架主入口
启动整个AI问答系统
"""

import asyncio
import mimetypes
import sys
from pathlib import Path
from typing import Optional, Iterable, Union
from src.preprocessing.preprocessor import Preprocessor
import os
import argparse
import textwrap
import sys
import time
from utils.handler import AttachmentHandler
from config import get_settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


from src.agent.orchestrator import AIAgent

settings = get_settings()

def parse_args():
    parser = argparse.ArgumentParser(description="智能代理框架")
    parser.add_argument('--text', type=str, help="查询文本")
    parser.add_argument('--attachments', type=str, help="附件路径", nargs='*')
    return parser.parse_args()

class IntelligentAgentApp:
    """智能代理应用程序"""

    def __init__(self):
        """初始化应用程序"""
        self.agent = AIAgent()
        self.preproc = Preprocessor(ocr_lang="eng+chi_sim")
        self.is_running = False
        api_key = settings.api_key
        self.handler = AttachmentHandler()

    async def start(self):
        """启动应用程序"""
        self.is_running = True

    async def stop(self):
        """停止应用程序"""
        print("🛑 The intelligent agent framework is being halted...")
        self.is_running = False
        print("✅The intelligent agent framework has been discontinued")

    async def process_query(
            self,
            text: Optional[str] = None,
            attachments: Optional[Iterable[Union[str, Path]]] = None,
            *,
            context: Optional[dict] = None,
    ) -> dict:
        if not self.is_running:
            return {
                'answer': 'The system has not been started yet. Please try again later.',
                'sources': [],
                'confidence': 0.0,
                'error': 'System not started'
            }
        ctx = context or {}

        try:
            pre = self.preproc  # Use the instance of Preprocessor initialized in the constructor

            preprocess_result = await asyncio.to_thread(
                pre.process,
                text or "",
                attachments,
            )

            user_input = {
                "raw_query": preprocess_result.raw_query,

                "processed_query": preprocess_result.processed_query,

                "attachments": [
                    {"path": str(x.path), "type": x.source_type, "content": x.content}
                    for x in (preprocess_result.pdf_attachments + preprocess_result.image_attachments)
                ],

                "attachment_issues": [i.model_dump() for i in preprocess_result.issues],
            }

            if user_input["attachments"]:
                self.agent.run(user_input)
                return await self.handler.process(
                    query=user_input["raw_query"],
                    attachments=user_input["attachments"]
                )
            else:
                result = self.agent.run(user_input)

            # 5) 把预处理的 issues 回填到返回值，方便 CLI 打印/上层可见
            result.setdefault("preprocess", {})
            result["preprocess"]["issues"] = user_input["attachment_issues"]

            return result
        except Exception as e:
            print(f"❌ An error occurred when processing the query: {e}")
            return {
                'answer': f'Sorry, I encountered an error when handling your issue: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }

    async def interactive_mode(self):
        """交互模式"""
        print("\n🤖 Enter the interactive mode, typing 'quit' or 'exit' to exit.")
        print("=" * 50)

        while self.is_running:
            try:
                query = input("\n👤 Please enter your question: ").strip()

                # 检查退出命令
                if query.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 Bye！")
                    break

                if not query:
                    print("❓ Please enter valid questions")
                    continue

                # 获取附件路径（用户可以输入附件路径）
                attachments_input = input("\nPlease Enter the attachment path (if any, separate multiple paths with Spaces and press Enter to skip) :").strip()

                # 如果附件输入不为空，处理附件路径
                attachments = [Path(att) for att in attachments_input.split()] if attachments_input else None

                # 处理查询
                print(f"\n🔄 Processing: {query}")
                result = await self.process_query(query, attachments=attachments)

                # 显示结果
                print(f"\n🤖 Response:")
                wrapped_lines = textwrap.wrap(result['answer'], width=60)

                for line in wrapped_lines:
                    for char in line:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        time.sleep(0.02)  # 控制打字速度，越小越快
                    sys.stdout.write('\n')  # 每行结束后换行

                print(f"\n📊 Confidence Coefficient: {result['confidence']:.2f}")

                if 'error' in result:
                    print(f"⚠️  Error Message: {result['error']}")

                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\n👋 Received an interrupt signal and is exiting...")
                break
            except Exception as e:
                print(f"\n❌ Something went wrong: {e}")


async def main():
    """主函数"""
    app = IntelligentAgentApp()

    try:
        # 启动应用
        await app.start()

        await app.interactive_mode()

    except KeyboardInterrupt:
        print("\nReceived an interrupt signal")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        await app.stop()


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 运行主函数
    asyncio.run(main())
