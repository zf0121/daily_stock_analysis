# -*- coding: utf-8 -*-
"""
===================================
AI 分析模块 - 适配股票与加密货币
===================================
"""
import logging
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import google.generativeai as genai
from datetime import datetime

logger = logging.getLogger(__name__)

# 定义 AI 返回的结构化数据格式
class AnalysisResult(BaseModel):
    code: str = Field(description="标的代码")
    name: str = Field(description="标的名称")
    operation_advice: str = Field(description="操作建议: 大力买入/建议买入/观望/建议卖出/坚决卖出")
    sentiment_score: int = Field(description="市场情绪评分 (0-100)")
    trend_prediction: str = Field(description="短期走势预测")
    risk_level: str = Field(description="风险等级: 低/中/高/极高")
    analysis_points: List[str] = Field(description="核心分析要点")
    technical_indicators: Dict[str, str] = Field(description="主要技术指标解读")
    summary: str = Field(description="一句话总结报告")

    def get_emoji(self) -> str:
        if "买入" in self.operation_advice: return "🚀"
        if "卖出" in self.operation_advice: return "⚠️"
        return "⚖️"

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"GeminiAnalyzer 初始化完成，使用模型: {model_name}")

    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None, 
                extra_context: str = "", is_crypto: bool = False) -> Optional[AnalysisResult]:
        """
        调用 AI 进行综合分析
        :param context: 技术面数据
        :param news_context: 搜索到的新闻/情报
        :param extra_context: 链上数据/情绪指数
        :param is_crypto: 是否为加密货币
        """
        try:
            if not self.api_key:
                logger.error("未配置 GEMINI_API_KEY")
                return None

            # 1. 构造系统角色和 Prompt
            if is_crypto:
                system_prompt = self._build_crypto_system_prompt(extra_context)
            else:
                system_prompt = self._build_stock_system_prompt()

            # 2. 构造用户数据 Prompt
            user_prompt = self._build_user_prompt(context, news_context, is_crypto)

            # 3. 调用 Gemini
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "response_mime_type": "application/json",
                }
            )

            full_prompt = f"{system_prompt}\n\n待分析数据如下：\n{user_prompt}\n\n请输出 JSON 格式结果。"
            response = model.generate_content(full_prompt)
            
            # 4. 解析结果
            result_dict = json.loads(response.text)
            return AnalysisResult(**result_dict)

        except Exception as e:
            logger.error(f"AI 分析发生异常: {e}")
            return None

    def _build_crypto_system_prompt(self, extra_context: str) -> str:
        return f"""
你是一位顶级的加密货币策略分析师，精通链上数据与技术面分析。
请根据提供的历史价格、成交量以及【市场情绪数据】进行深度研判。

【核心原则】：
1. 波动性：加密货币波动巨大，请给出更具容错空间的建议。
2. 情绪驱动：高度参考恐慌贪婪指数。
3. 禁忌：不要提到市盈率、财报、法人等股票术语。

【当前市场参考】：
{extra_context}

请严格按 JSON 格式输出包含：code, name, operation_advice, sentiment_score, trend_prediction, risk_level, analysis_points, technical_indicators, summary。
"""

    def _build_stock_system_prompt(self) -> str:
        return """
你是一位资深的 A 股证券分析师，擅长量价分析和筹码分布研究。
请基于技术面和最新情报给出专业、客观的投资建议。
请严格按 JSON 格式输出。
"""

    def _build_user_prompt(self, context: Dict[str, Any], news_context: Optional[str], is_crypto: bool) -> str:
        # 提取关键指标
        name = context.get('stock_name', '未知')
        code = context.get('code', '未知')
        realtime = context.get('realtime', {})
        trend = context.get('trend_analysis', {})
        
        # 基础量价信息
        prompt = f"""
标的名称：{name} ({code})
最新价格：{realtime.get('price', 'N/A')}
量比/换手：{realtime.get('volume_ratio', 'N/A')} / {realtime.get('turnover_rate', 'N/A')}%
趋势状态：{trend.get('trend_status', 'N/A')}
买入信号评分：{trend.get('signal_score', 'N/A')}
"""
        # 添加技术面细节
        if 'chip' in context:
            prompt += f"筹码获利比：{context['chip'].get('profit_ratio', 'N/A')}\n"

        # 添加新闻/情报
        if news_context:
            prompt += f"\n【最新相关情报】:\n{news_context}\n"
        
        return prompt

# 为了兼容性，保留原有的映射逻辑（可选）
STOCK_NAME_MAP = {}
