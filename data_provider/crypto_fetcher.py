import yfinance as yf
import pandas as pd
import requests
import logging
from typing import Optional

# 配置日志
logger = logging.getLogger(__name__)

class CryptoFetcher:
    def __init__(self):
        self.fng_url = "https://api.alternative.me/fng/"

    def get_crypto_data(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """
        获取加密货币K线数据并标准化
        :param symbol: 交易对名称，如 'BTC-USD', 'ETH-USD'
        :param days: 获取天数
        """
        try:
            logger.info(f"正在从 yfinance 获取 {symbol} 的K线数据...")
            # 获取数据，虚拟货币 7x24 交易，无需考虑开盘时间
            df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
            
            if df.empty:
                logger.warning(f"{symbol} 未能获取到数据，请检查代码拼写是否正确（如 BTC-USD）。")
                return None

            # 重置索引，将 Date 变为一列
            df = df.reset_index()
            
            # 处理 yfinance 可能返回的多级表头 (MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 统一列名为小写，适配原项目的 analyzer 逻辑
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # 将日期转换为字符串 YYYY-MM-DD
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # 计算技术指标所需的基础列
            df['change'] = df['close'].diff()
            df['pct_change'] = df['close'].pct_change() * 100
            
            # 填充第一行的 NaN
            df = df.fillna(0)
            
            return df

        except Exception as e:
            logger.error(f"获取 {symbol} K线数据时发生异常: {str(e)}")
            return None

    def get_onchain_sentiment(self) -> str:
        """
        获取加密货币特有的市场情绪面数据（链上及情绪指标）
        """
        sentiment_report = "\n【加密货币专项参考数据】\n"
        
        # 1. 获取恐慌与贪婪指数 (Fear & Greed Index)
        try:
            response = requests.get(self.fng_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                fng_val = data['data'][0]['value']
                fng_class = data['data'][0]['value_classification']
                sentiment_report += f"- 恐慌贪婪指数: {fng_val} ({fng_class})\n"
                
                # 为 AI 提供背景知识建议
                if int(fng_val) < 25:
                    sentiment_report += "  (提示: 市场处于极度恐慌状态，通常是潜在的左侧买点)\n"
                elif int(fng_val) > 75:
                    sentiment_report += "  (提示: 市场处于极度贪婪状态，需警惕回调风险)\n"
        except Exception as e:
            logger.error(f"获取情绪指数失败: {e}")
            sentiment_report += "- 恐慌贪婪指数: 暂时无法获取\n"

        # 2. 获取大盘参考指标 (以 BTC 为基准)
        try:
            btc = yf.Ticker("BTC-USD")
            # 尝试获取市值等信息 (GitHub Actions 运行环境 IP 有时会被限制获取 info)
            # 我们通过 history 获取最近两天的收盘价计算简单趋势
            hist = btc.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[0]
                curr_close = hist['Close'].iloc[1]
                trend = "上涨" if curr_close > prev_close else "下跌"
                sentiment_report += f"- BTC大盘24h走势: {trend}\n"
        except Exception:
            pass

        return sentiment_report

    def get_realtime_price(self, symbol: str) -> float:
        """获取当前最新实时价格"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            return data['Close'].iloc[-1] if not data.empty else 0.0
        except Exception:
            return 0.0
