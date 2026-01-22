# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import get_config, Config
from storage import get_db
from data_provider import DataFetcherManager
from data_provider.crypto_fetcher import CryptoFetcher
from data_provider.akshare_fetcher import AkshareFetcher
from analyzer import GeminiAnalyzer, AnalysisResult, STOCK_NAME_MAP
from search_service import SearchService
from stock_analyzer import StockTrendAnalyzer
from market_analyzer import MarketAnalyzer

# === 兼容性处理：尝试导入通知组件 ===
try:
    from notification import NotificationService, NotificationChannel, send_daily_report
except ImportError:
    # 如果基础通知组件都导入失败，说明项目结构有问题
    raise ImportError("无法从 notification.py 导入基础组件，请检查该文件是否存在。")

try:
    from notification import FeishuDocManager
    HAS_FEISHU_DOC = True
except ImportError:
    # 如果没有 FeishuDocManager，我们标记为不可用，但不报错
    HAS_FEISHU_DOC = False
    class FeishuDocManager: # 定义一个空类防止后续引用报错
        def is_configured(self): return False

# 配置日志
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(level)

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.crypto_fetcher = CryptoFetcher()
        self.akshare_fetcher = AkshareFetcher()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=self.config.tavily_api_keys,
            serpapi_keys=self.config.serpapi_keys,
        )

    def is_crypto(self, code: str) -> bool:
        return any(c.isalpha() for c in code)

    def fetch_and_save_stock_data(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            if self.is_crypto(code):
                df = self.crypto_fetcher.get_crypto_data(code)
                source = "yfinance"
            else:
                df, source = self.fetcher_manager.get_daily_data(code, days=30)
            if df is None or df.empty: return False, "数据为空"
            self.db.save_daily_data(df, code, source)
            return True, None
        except Exception as e: return False, str(e)

    def analyze_stock(self, code: str) -> Optional[AnalysisResult]:
        try:
            is_crypto_asset = self.is_crypto(code)
            stock_name = STOCK_NAME_MAP.get(code, code)
            extra_data = self.crypto_fetcher.get_onchain_sentiment() if is_crypto_asset else ""
            
            context = self.db.get_analysis_context(code)
            if not context: return None
            
            # AI 分析
            return self.analyzer.analyze(context, extra_context=extra_data, is_crypto=is_crypto_asset)
        except Exception as e:
            logger.error(f"[{code}] 分析异常: {e}")
            return None

    def run(self, stock_codes: Optional[List[str]] = None):
        if stock_codes is None: stock_codes = self.config.stock_list
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_task, code): code for code in stock_codes}
            for f in as_completed(futures):
                res = f.result()
                if res: results.append(res)
        
        if results and self.notifier.is_available():
            report = self.notifier.generate_dashboard_report(results)
            self.notifier.send(report)
        return results

    def process_task(self, code):
        success, _ = self.fetch_and_save_stock_data(code)
        return self.analyze_stock(code) if success else None

def run_full_analysis(config: Config, args: argparse.Namespace, stock_codes: Optional[List[str]] = None):
    pipeline = StockAnalysisPipeline(config=config)
    results = pipeline.run(stock_codes=stock_codes)
    
    # 尝试运行大盘复盘
    if config.market_review_enabled:
        try:
            ma = MarketAnalyzer(search_service=pipeline.search_service, analyzer=pipeline.analyzer)
            report = ma.run_daily_review()
            if report: pipeline.notifier.send(f"🎯 大盘复盘\n\n{report}")
        except: pass

    # 飞书文档（仅在支持时运行）
    if HAS_FEISHU_DOC:
        try:
            feishu = FeishuDocManager()
            if feishu.is_configured():
                feishu.create_daily_doc("投资复盘", "内容...")
        except: pass

def main():
    args = argparse.Namespace(stocks=None, debug=False, dry_run=False, no_notify=False, workers=3, schedule=False, no_market_review=False)
    config = get_config()
    setup_logging(debug=args.debug)
    run_full_analysis(config, args)

if __name__ == "__main__":
    main()
