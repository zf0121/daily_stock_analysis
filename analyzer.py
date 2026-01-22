# -*- coding: utf-8 -*-
import os
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import google.generativeai as genai
from datetime import datetime

logger = logging.getLogger(__name__)

# 保持你原始定义的 AnalysisResult 不变
class AnalysisResult(BaseModel):
    code: str
    name: str
    operation_advice: str  # 大力买入/建议买入/观望/建议卖出/坚决卖出
    sentiment_score: int    # 0-100
    trend_prediction: str
    risk_level: str        # 低/中/高/极高
    analysis_points: List[str]
    technical_indicators: Dict[str, str]
    summary: str
    
    def get_emoji(self) -> str:
        if "买入" in self.operation_advice: return "🚀"
        if "卖出" in self.operation_advice: return "⚠️"
        return "⚖️"

# 映射表（保持原样）
STOCK_NAME_MAP = {
    "sh600519": "贵州茅台",
    "sh601318": "中国平安",
    "sz000001": "平安银行",
    "sz000725": "京东方A",
    "sz002415": "海康威视"
}

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"GeminiAnalyzer 初始化完成，使用模型: {model_name}")

    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None, 
                extra_context: str = "", is_crypto: bool = False) -> Optional[AnalysisResult]:
        """综合分析核心函数"""
        try:
            if not self.api_key:
                logger.error("未配置 GEMINI_API_KEY")
                return None

            # 1. 区分资产类型构建系统 Prompt
            if is_crypto:
                system_prompt = self._build_crypto_system_prompt(extra_context)
            else:
                system_prompt = self._build_stock_system_prompt()

            # 2. 构建用户数据部分
            user_prompt = self._build_user_prompt(context, news_context)

            # 3. 调用 AI
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.2, # 降低随机性，保证 JSON 稳定
                    "top_p": 0.8,
                    "response_mime_type": "application/json",
                }
            )

            full_content = f"{system_prompt}\n\n待分析数据如下：\n{user_prompt}\n\n请输出符合格式的 JSON 结果。"
            response = model.generate_content(full_content)
            
            # 4. 解析 JSON
            return self._safe_parse_response(response.text)

        except Exception as e:
            logger.error(f"AI 分析失败: {str(e)}")
            return None

    def _build_crypto_system_prompt(self, extra_context: str) -> str:
        """加密货币专用 Prompt"""
        return f"""你是一位全球顶尖的加密货币量化交易员。
请基于技术面数据和链上情绪进行分析。
注意：加密货币 7x24 交易，波动大。请结合以下【链上/情绪数据】综合判断：
{extra_context}

必须输出 JSON 格式，字段包含：code, name, operation_advice, sentiment_score, trend_prediction, risk_level, analysis_points, technical_indicators, summary。
不要提及 A 股、财报、市盈率等概念。"""

    def _build_stock_system_prompt(self) -> str:
        """A股专用 Prompt (复刻你原始 1223 行代码中的核心逻辑)"""
        return """你是一位深耕 A 股多年的资深首席分析师。
请结合量价关系、筹码分布、均线系统进行深度复盘。
必须输出 JSON 格式，字段包含：code, name, operation_advice, sentiment_score, trend_prediction, risk_level, analysis_points, technical_indicators, summary。"""

    def _build_user_prompt(self, context: Dict[str, Any], news_context: Optional[str]) -> str:
        """通用的数据组装逻辑"""
        name = context.get('stock_name', '未知标的')
        code = context.get('code', '未知代码')
        
        # 提取各个维度的详细数据（适配 main.py 传过来的字典）
        realtime = context.get('realtime', {})
        chip = context.get('chip', {})
        trend = context.get('trend_analysis', {})
        
        prompt = f"""
分析对象：{name} ({code})
---
【技术面信息】
当前价格: {realtime.get('price', '数据缺失')}
量比: {realtime.get('volume_ratio', '数据缺失')}
换手率: {realtime.get('turnover_rate', '数据缺失')}%
趋势状态: {trend.get('trend_status', '数据缺失')}
信号得分: {trend.get('signal_score', '数据缺失')}
筹码获利比: {chip.get('profit_ratio', '数据缺失')}

【市场情报】
{news_context if news_context else "暂无关键新闻"}
"""
        return prompt

    def _safe_parse_response(self, text: str) -> Optional[AnalysisResult]:
        """安全解析 JSON"""
        try:
            # 清理可能的 Markdown 标记
            clean_json = re.sub(r'```json\n?|\n?```', '', text).strip()
            data = json.loads(clean_json)
            return AnalysisResult(**data)
        except Exception as e:
            logger.error(f"JSON 解析失败: {e}, 原始内容: {text}")
            return None
