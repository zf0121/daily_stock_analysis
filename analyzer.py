# -*- coding: utf-8 -*-
"""
===================================
A股 & Crypto 智能分析层 (兼容版)
===================================
"""
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

# --- 1. 定义数据结构 (必须保留，main.py 需要用到) ---
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

# --- 2. 常用股票映射 (保留以防 main.py 引用) ---
STOCK_NAME_MAP = {
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    'BTC-USD': '比特币',
    'ETH-USD': '以太坊'
}

# --- 3. 分析器核心类 ---
class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("未配置 GEMINI_API_KEY，AI 分析功能将不可用")
        else:
            genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        
        # 配置生成参数
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "response_mime_type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None, 
                extra_context: str = "", is_crypto: bool = False) -> Optional[AnalysisResult]:
        """
        执行 AI 分析
        :param context: 包含技术指标、价格等数据的字典
        :param news_context: 新闻搜索结果字符串
        :param extra_context: (新增) 额外数据，如恐慌指数
        :param is_crypto: (新增) 是否为加密货币
        """
        try:
            if not self.api_key:
                return None

            # 1. 动态选择 System Prompt (关键修复点)
            if is_crypto:
                system_prompt = self._build_crypto_prompt(extra_context)
            else:
                system_prompt = self._build_stock_prompt()

            # 2. 构建用户输入
            user_prompt = self._build_user_prompt(context, news_context)

            # 3. 初始化模型
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )

            # 4. 发送请求
            full_prompt = f"{system_prompt}\n\n【待分析数据】\n{user_prompt}\n\n请严格输出 JSON。"
            response = model.generate_content(full_prompt)
            
            # 5. 解析结果
            return self._parse_response(response.text)

        except Exception as e:
            logger.error(f"AI 分析过程发生错误: {e}")
            raise e # 抛出异常以触发 retry

    def _build_crypto_prompt(self, extra_context: str) -> str:
        """生成加密货币专用的 System Prompt"""
        return f"""你是一位专业的加密货币交易策略师。
请基于提供的K线数据、技术指标以及市场情绪进行分析。

【特别注意】：
1. Crypto 市场 7x24 小时交易，无涨跌停。
2. 重点关注：MA 均线趋势、成交量变化、RSI 超买超卖。
3. 必须参考以下【链上/情绪数据】：
{extra_context}

请输出纯 JSON 格式，包含字段：code, name, operation_advice, sentiment_score, trend_prediction, risk_level, analysis_points, technical_indicators, summary。
"""

    def _build_stock_prompt(self) -> str:
        """生成 A 股专用的 System Prompt"""
        return """你是一位资深的 A 股证券分析师。
请结合量价关系、筹码分布、均线系统对股票进行深度复盘。
分析逻辑：
1. 趋势优先：判断长期和短期均线排列。
2. 筹码为王：关注获利盘比例。
3. 风险控制：给出明确的止盈止损建议。

请输出纯 JSON 格式，包含字段：code, name, operation_advice, sentiment_score, trend_prediction, risk_level, analysis_points, technical_indicators, summary。
"""

    def _build_user_prompt(self, context: Dict[str, Any], news_context: Optional[str]) -> str:
        """组装用户输入数据"""
        name = context.get('stock_name', '未知标的')
        code = context.get('code', '未知代码')
        
        # 安全获取数据，防止 KeyError
        realtime = context.get('realtime', {})
        chip = context.get('chip', {})
        trend = context.get('trend_analysis', {})
        
        prompt = f"""
标的信息：{name} ({code})
---
【量价数据】
现价: {realtime.get('price', 'N/A')}
量比: {realtime.get('volume_ratio', 'N/A')}
换手率: {realtime.get('turnover_rate', 'N/A')}%

【技术信号】
趋势状态: {trend.get('trend_status', 'N/A')}
买入评分: {trend.get('signal_score', 'N/A')}
筹码获利比: {chip.get('profit_ratio', 'N/A')}

【市场情报】
{news_context if news_context else "暂无特殊情报"}
"""
        return prompt

    def _parse_response(self, text: str) -> Optional[AnalysisResult]:
        """解析 AI 返回的 JSON"""
        try:
            # 去除可能的 Markdown 代码块标记
            clean_text = re.sub(r'```json\n?|\n?```', '', text).strip()
            data = json.loads(clean_text)
            return AnalysisResult(**data)
        except json.JSONDecodeError:
            logger.error(f"JSON 解析失败，AI 返回内容: {text}")
            return None
        except Exception as e:
            logger.error(f"结果转换失败: {e}")
            return None
