from typing import Optional
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from chatchat.settings import Settings

class LLMApi:
    _instance = None
    _llm: Optional[BaseLLM] = None
    
    def __init__(self):
        raise RuntimeError("请使用 get_llm() 获取 LLM 实例")
    
    @classmethod
    def initialize(cls) -> BaseLLM:
        """初始化 LLM 实例"""
        if not cls._llm:
            # 从配置中获取 LLM 相关设置
            model_name = Settings.llm_settings.MODEL_NAME
            api_base_url = Settings.llm_settings.API_BASE_URL
            api_key = Settings.llm_settings.API_KEY
            
            # 创建 LLM 实例
            cls._llm = ChatOpenAI(
                model_name=model_name,
                openai_api_base=api_base_url,
                openai_api_key=api_key,
                temperature=0.1,  # 降低随机性，使分块更稳定
                request_timeout=60,
            )
        
        return cls._llm

def get_llm() -> BaseLLM:
    """获取 LLM 实例的全局访问点"""
    return LLMApi.initialize()

# 示例用法
# if __name__ == "__main__":
    # llm = get_llm()
    # response = llm.predict("请将下面的文本分成两个语义完整的段落：\n\n"
    #                       "人工智能是一个广泛的领域，包括机器学习、深度学习等多个分支。"
    #                       "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习和改进。")
    # print(response)