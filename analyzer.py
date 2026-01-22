# -*- coding: utf-8 -*-
import os
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# --- 数据结构 ---
class AnalysisResult(BaseModel):
    code: str
    name: str
    operation_advice: str
    sentiment_score: int
    trend_prediction: str
    risk_level: str
    analysis_points: List[str]
    technical_indicators: Dict[str, str]
    summary: str
    
    def get_emoji(self) -> str:
        if "买入" in self.operation_advice: return "🚀"
        if "卖出" in self.operation_advice: return "⚠️"
        return "⚖️"

STOCK_NAME_MAP = {'BTC-USD': '比特币', 'ETH-USD': '以太坊'}

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key: genai.configure(api_key=self.api_key)
        self.model_name = model_name

    # === 关键修改：加入 **kwargs 忽略多余参数，防止报错 ===
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None, 
                extra_context: str = "", is_crypto: bool = False, **kwargs) -> Optional[AnalysisResult]:
        try:
            if not self.api_key: return None

            # 1. 简易 Prompt 构建
            if is_crypto:
                sys_prompt = f"你是加密货币专家。请分析以下数据。链上参考：{extra_context}。请输出JSON。"
            else:
                sys_prompt = "你是A股专家。请分析以下数据。请输出JSON。"

            # 2. 用户 Prompt
            user_prompt = f"分析对象：{context.get('stock_name')} \n数据：{context} \n情报：{news_context}"

            # 3. 调用 AI
            model = genai.GenerativeModel(self.model_name, generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(f"{sys_prompt}\n\n{user_prompt}")
            
            # 4. 解析
            clean_text = re.sub(r'```json\n?|\n?```', '', response.text).strip()
            return AnalysisResult(**json.loads(clean_text))

        except Exception as e:
            logger.error(f"分析出错: {e}")
            return None
