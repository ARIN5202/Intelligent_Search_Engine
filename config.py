"""
配置管理模块
从环境变量和.env文件中读取配置
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


def _load_env_file():
    """加载.env文件"""
    env_file = Path(__file__).parent / '.env'

    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # 只有在环境变量中不存在时才设置
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"加载.env文件失败: {e}")
    else:
        print("未找到.env文件，将使用默认配置")


def get(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    获取配置值

    Args:
        key: 配置项名称
        default: 默认值

    Returns:
        配置值或默认值
    """
    return os.environ.get(key, default)


def get_int(key: str, default: int = 0) -> int:
    """
    获取整数配置值

    Args:
        key: 配置项名称
        default: 默认值

    Returns:
        整数配置值
    """
    value = get(key)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        print(f"配置项 {key} 的值 '{value}' 不是有效整数，使用默认值 {default}")
        return default


def get_bool(key: str, default: bool = False) -> bool:
    """
    获取布尔配置值

    Args:
        key: 配置项名称
        default: 默认值

    Returns:
        布尔配置值
    """
    value = get(key, '').lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    elif value in ('false', '0', 'no', 'off'):
        return False
    else:
        return default


def get_list(key: str, separator: str = ',', default: Optional[list] = None) -> list:
    """
    获取列表配置值

    Args:
        key: 配置项名称
        separator: 分隔符
        default: 默认值

    Returns:
        列表配置值
    """
    value = get(key)
    if value is None:
        return default or []

    return [item.strip() for item in value.split(separator) if item.strip()]


def _validate_config():
    """验证必要的配置项"""
    required_keys = [
        'OPENAI_API_KEY',
        'SEARCH_API_KEY'
    ]

    missing_keys = []
    for key in required_keys:
        if not get(key):
            missing_keys.append(key)

    if missing_keys:
        print(f"警告：缺少以下配置项: {', '.join(missing_keys)}")
        print("某些功能可能无法正常工作，请在.env文件中配置相应的API密钥")


class Config:
    """配置管理器"""

    def __init__(self):
        """初始化配置"""
        _load_env_file()
        _validate_config()

    # 便捷方法获取常用配置
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API密钥"""
        return get('OPENAI_API_KEY')

    @property
    def openai_base_url(self) -> str:
        """OpenAI API基础URL"""
        return get('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    @property
    def search_api_key(self) -> Optional[str]:
        """搜索API密钥"""
        return get('SEARCH_API_KEY')

    @property
    def weather_api_key(self) -> Optional[str]:
        """天气API密钥"""
        return get('WEATHER_API_KEY')

    @property
    def finance_api_key(self) -> Optional[str]:
        """金融API密钥"""
        return get('FINANCE_API_KEY')

    @property
    def maps_api_key(self) -> Optional[str]:
        """地图API密钥"""
        return get('MAPS_API_KEY')

    @property
    def vector_db_url(self) -> str:
        """向量数据库URL"""
        return get('VECTOR_DB_URL', 'localhost:6333')

    @property
    def log_level(self) -> str:
        """日志级别"""
        return get('LOG_LEVEL', 'INFO')

    @property
    def max_retrieval_results(self) -> int:
        """最大检索结果数"""
        return get_int('MAX_RETRIEVAL_RESULTS', 50)

    @property
    def default_top_k(self) -> int:
        """默认返回结果数"""
        return get_int('DEFAULT_TOP_K', 10)

    @property
    def response_timeout(self) -> int:
        """响应超时时间（秒）"""
        return get_int('RESPONSE_TIMEOUT', 30)

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            'openai_api_key': self.openai_api_key,
            'openai_base_url': self.openai_base_url,
            'search_api_key': self.search_api_key,
            'weather_api_key': self.weather_api_key,
            'finance_api_key': self.finance_api_key,
            'maps_api_key': self.maps_api_key,
            'vector_db_url': self.vector_db_url,
            'log_level': self.log_level,
            'max_retrieval_results': self.max_retrieval_results,
            'default_top_k': self.default_top_k,
            'response_timeout': self.response_timeout
        }

    def __str__(self) -> str:
        """字符串表示（隐藏敏感信息）"""
        config_dict = self.to_dict()
        # 隐藏API密钥
        for key in config_dict:
            if 'key' in key.lower() and config_dict[key]:
                config_dict[key] = f"{config_dict[key][:8]}..."

        return f"Config({config_dict})"


# 全局配置实例
config = Config()