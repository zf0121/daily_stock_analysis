# -*- coding: utf-8 -*-
import os
import json
import logging
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# --- 核心数据模型 ---
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

STOCK_NAME_MAP = {}

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name

    # 使用 **kwargs 确保即使 main.py 传了乱七八糟的参数也不会崩溃
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None, 
                extra_context: str = "", is_crypto: bool = False, **kwargs) -> Optional[AnalysisResult]:
        try:
            if not self.api_key:
                logger.error("API KEY 缺失")
                return None

            # 自动切换 Prompt
            if is_crypto:
                system_prompt = f"你是一个加密货币专家。参考情绪：{extra_context}。请输出JSON分析。"
            else:
                system_prompt = "你是一个A股分析专家。请输出JSON分析。"

            user_data = f"标的：{context.get('stock_name')}，数据：{context}，情报：{news_context}"
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"response_mime_type": "application/json", "temperature": 0.2}
            )

            response = model.generate_content(f"{system_prompt}\n\n数据：{user_data}")
            
            # 安全解析
            res_text = response.text
            clean_json = re.sub(r'```json\n?|\n?```', '', res_text).strip()
            return AnalysisResult(**json.loads(clean_json))

        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            return None

def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer()
