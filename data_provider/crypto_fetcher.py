# data_provider/crypto_fetcher.py

import yfinance as yf
import pandas as pd
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class CryptoFetcher:
    def __init__(self):
        pass

    def get_crypto_data(self, symbol: str, days: int = 100) -> Optional[pd.DataFrame]:
        """
        获取加密货币数据并转换为项目通用的 DataFrame 格式
        :param symbol: 交易对代码，例如 'BTC-USD', 'ETH-USD'
        :param days: 获取最近多少天的数据
        """
        try:
            # yfinance 获取数据
            # 虚拟货币 7x24，我们获取足够长的数据以计算均线
            df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
            
            if df.empty:
                logger.warning(f"{symbol} 未获取到数据")
                return None

            # yfinance 的索引是 Date，需要重置索引并重命名列以匹配原有 AkShare 的格式
            df = df.reset_index()
            
            # 扁平化多层索引 (如果 yfinance 返回了多层列名)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 标准化列名（原有项目通常使用小写）
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # 确保日期格式为字符串 YYYY-MM-DD (原有逻辑通常依赖字符串日期)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # 计算涨跌幅 (pct_change) 和 涨跌额 (change)
            # 股市数据通常包含这些，yfinance 需要手动计算
            df['change'] = df['close'].diff()
            df['pct_change'] = df['close'].pct_change() * 100
            
            # 填补 NaN
            df = df.fillna(0)

            return df

        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None

    def get_realtime_price(self, symbol: str) -> float:
        """获取最新价格"""
        try:
            ticker = yf.Ticker(symbol)
            # 获取最新的一分钟数据
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
            return 0.0
        except Exception:
            return 0.0
try:
            df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
            if df.empty: return None
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df['change'] = df['close'].diff()
            df['pct_change'] = df['close'].pct_change() * 100
            return df.fillna(0)
        except Exception as e:
            logger.error(f"K线数据获取失败: {e}")
            return None

    def get_onchain_sentiment(self):
        """
        获取加密货币市场情绪和基础数据
        返回一个字符串，可以直接喂给 AI
        """
        info_text = ""
        
        # 1. 获取恐慌贪婪指数 (免费 API)
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            if data['metadata']['error'] is None:
                fng_value = data['data'][0]['value']
                fng_class = data['data'][0]['value_classification']
                info_text += f"【市场情绪】当前恐慌贪婪指数为 {fng_value} ({fng_class})。\n"
                
                # 简单逻辑判断供 AI 参考
                if int(fng_value) < 20:
                    info_text += "注：市场处于极度恐慌，历史上通常是阶段性底部区域。\n"
                elif int(fng_value) > 80:
                    info_text += "注：市场处于极度贪婪，历史上需警惕回调风险。\n"
        except Exception as e:
            logger.error(f"恐慌指数获取失败: {e}")

        # 2. 获取比特币市占率 (通过 yfinance 简易获取)
        # 很多时候 BTC.D (市占率) 上涨意味着吸血行情，山寨币会跌
        try:
            # 获取 BTC 市值和 ETH 市值来做一个简单的对比
            btc_ticker = yf.Ticker("BTC-USD")
            btc_info = btc_ticker.info
            # 注意：Github Actions IP 可能获取不到详细 info，如果获取不到会跳过
            if 'marketCap' in btc_info and btc_info['marketCap']:
                mcap = btc_info['marketCap'] / 1000000000 # 转为十亿美元
                vol24 = btc_info.get('volume24Hr', 0) / 1000000000
                info_text += f"【链上/市场数据】BTC市值: ${mcap:.2f}B, 24h交易量: ${vol24:.2f}B。\n"
        except Exception:
            pass # 忽略错误

        return info_text
