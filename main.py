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
        self.handler = AttachmentHandler(api_key=api_key)

    async def start(self):
        """启动应用程序"""
        self.is_running = True

    async def stop(self):
        """停止应用程序"""
        print("🛑 正在停止智能代理框架...")
        self.is_running = False
        print("✅ 智能代理框架已停止")

    async def process_query(
            self,
            text: Optional[str] = None,
            attachments: Optional[Iterable[Union[str, Path]]] = None,
            *,
            context: Optional[dict] = None,
    ) -> dict:
        if not self.is_running:
            return {
                'answer': '系统尚未启动，请稍候再试。',
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
            print(f"❌ 处理查询时发生错误: {e}")
            return {
                'answer': f'抱歉，处理您的问题时遇到了错误: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }

    async def interactive_mode(self):
        """交互模式"""
        print("\n🤖 进入交互模式，输入 'quit' 或 'exit' 退出")
        print("=" * 50)

        while self.is_running:
            try:
                query = input("\n👤 请输入您的问题: ").strip()

                # 检查退出命令
                if query.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！")
                    break

                if not query:
                    print("❓ 请输入有效的问题")
                    continue

                # 获取附件路径（用户可以输入附件路径）
                attachments_input = input("\n请输入附件路径（如果有的话，多个路径用空格分隔，按 Enter 跳过）：").strip()

                # 如果附件输入不为空，处理附件路径
                attachments = [Path(att) for att in attachments_input.split()] if attachments_input else None

                # 处理查询
                print(f"\n🔄 正在处理: {query}")
                result = await self.process_query(query, attachments=attachments)

                # 显示结果
                print(f"\n🤖 回答:")
                wrapped_lines = textwrap.wrap(result['answer'], width=70)

                for line in wrapped_lines:
                    for char in line:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        time.sleep(0.02)  # 控制打字速度，越小越快
                    sys.stdout.write('\n')  # 每行结束后换行

                if result['sources']:
                    print(f"\n📚 参考来源:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"{i}. {source['title']} (评分: {source['score']:.2f})")

                print(f"\n📊 置信度: {result['confidence']:.2f}")

                if 'error' in result:
                    print(f"⚠️  错误信息: {result['error']}")

                print("-" * 50)

            except KeyboardInterrupt:
                print("\n\n👋 收到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")


async def main():
    """主函数"""
    app = IntelligentAgentApp()

    try:
        # 启动应用
        await app.start()

        await app.interactive_mode()

    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"应用程序错误: {e}")
    finally:
        await app.stop()


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 运行主函数
    asyncio.run(main())
