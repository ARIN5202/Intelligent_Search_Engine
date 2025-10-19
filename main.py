#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能代理框架主入口
启动整个AI问答系统
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import config
from src.agent.orchestrator import AIAgent


class IntelligentAgentApp:
    """智能代理应用程序"""

    def __init__(self):
        """初始化应用程序"""
        self.agent = AIAgent()
        self.is_running = False

    async def start(self):
        """启动应用程序"""
        print("🚀 智能代理框架启动中...")
        print(f"📋 配置信息: {config}")

        # 健康检查
        health_status = await self.agent.health_check()
        print(f"🏥 组件健康状态: {health_status}")

        # 检查是否有组件不健康
        unhealthy_components = [comp for comp, status in health_status.items() if not status]
        if unhealthy_components:
            print(f"⚠️  警告：以下组件不健康: {unhealthy_components}")
            print("某些功能可能受到影响")

        self.is_running = True
        print("✅ 智能代理框架启动完成！")

    async def stop(self):
        """停止应用程序"""
        print("🛑 正在停止智能代理框架...")
        self.is_running = False
        print("✅ 智能代理框架已停止")

    async def process_query(self, query: str, context: Optional[dict] = None) -> dict:
        """
        处理用户查询

        Args:
            query: 用户问题
            context: 可选的上下文信息

        Returns:
            处理结果
        """
        if not self.is_running:
            return {
                'answer': '系统尚未启动，请稍候再试。',
                'sources': [],
                'confidence': 0.0,
                'error': 'System not started'
            }

        try:
            # Create the proper user_input dictionary expected by AIAgent.run()
            user_input = {
                "query": query,
                "images": context.get("images", []) if context else []
            }
            
            # Pass the dictionary to the agent
            result = self.agent.run(user_input)
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
                # 获取用户输入
                query = input("\n👤 请输入您的问题: ").strip()

                # 检查退出命令
                if query.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 再见！")
                    break

                if not query:
                    print("❓ 请输入有效的问题")
                    continue

                # 处理查询
                print(f"\n🔄 正在处理: {query}")
                result = await self.process_query(query)

                # 显示结果
                print(f"\n🤖 回答:")
                print(f"{result['answer']}")

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

    async def demo_mode(self):
        """演示模式"""
        print("\n🎯 演示模式：展示系统功能")
        print("=" * 50)

        demo_queries = [
            "公司的考勤制度是什么？",
            "北京今天的天气怎么样？",
            "苹果公司的股票价格",
            "从天安门到故宫怎么走？",
            "人工智能的最新发展趋势"
        ]

        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 演示查询 {i}: {query}")
            result = await self.process_query(query)

            print(f"🤖 回答: {result['answer'][:200]}...")
            print(f"📊 置信度: {result['confidence']:.2f}")
            print(f"📚 来源数量: {len(result['sources'])}")

            # 等待一下再继续
            await asyncio.sleep(1)

        print("\n✅ 演示完成！")


async def main():
    """主函数"""
    app = IntelligentAgentApp()

    try:
        # 启动应用
        await app.start()

        # 检查命令行参数
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()

            if mode == 'demo':
                # 演示模式
                await app.demo_mode()
            elif mode == 'query' and len(sys.argv) > 2:
                # 单次查询模式
                query = ' '.join(sys.argv[2:])
                result = await app.process_query(query)
                print(f"问题: {query}")
                print(f"回答: {result['answer']}")
                print(f"置信度: {result['confidence']:.2f}")
            else:
                print("用法:")
                print("  python main.py            # 交互模式")
                print("  python main.py demo       # 演示模式")
                print("  python main.py query 你的问题  # 单次查询")
        else:
            # 默认交互模式
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