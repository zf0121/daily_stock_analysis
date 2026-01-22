# -*- coding: utf-8 -*-
"""
===================================
自选股 & 加密货币智能分析系统
===================================
修改点：
1. 集成 CryptoFetcher，支持 BTC/ETH 等虚拟货币分析
2. 自动识别股票与加密货币代码
3. 注入链上情绪数据（恐慌贪婪指数）
"""
import os

# 代理配置 - 仅在本地环境使用
if os.getenv("GITHUB_ACTIONS") != "true":
    pass

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import get_config, Config
from storage import get_db, DatabaseManager
from data_provider import DataFetcherManager
# 导入新增的 CryptoFetcher
from data_provider.crypto_fetcher import CryptoFetcher
from data_provider.akshare_fetcher import AkshareFetcher, RealtimeQuote, ChipDistribution
from analyzer import GeminiAnalyzer, AnalysisResult, STOCK_NAME_MAP
from notification import NotificationService, NotificationChannel, send_daily_report
from search_service import SearchService, SearchResponse
from enums import ReportType
from stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from market_analyzer import MarketAnalyzer

# 配置日志格式
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    level = logging.DEBUG if debug else logging.INFO
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"analysis_{today_str}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)
    
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.crypto_fetcher = CryptoFetcher()  # 初始化加密货币抓取器
        self.akshare_fetcher = AkshareFetcher()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=self.config.tavily_api_keys,
            serpapi_keys=self.config.serpapi_keys,
        )
        logger.info(f"调度器初始化完成，支持股票与加密货币双模分析")

    def is_crypto(self, code: str) -> bool:
        """判断是否为加密货币代码 (包含字母即视为加密货币)"""
        return any(c.isalpha() for c in code)

    def fetch_and_save_stock_data(self, code: str, force_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        try:
            today = date.today()
            if not force_refresh and self.db.has_today_data(code, today):
                return True, None
            
            if self.is_crypto(code):
                logger.info(f"[{code}] 识别为加密货币，从 yfinance 获取数据...")
                df = self.crypto_fetcher.get_crypto_data(code)
                source_name = "yfinance"
            else:
                logger.info(f"[{code}] 识别为股票，从 AkShare 获取数据...")
                df, source_name = self.fetcher_manager.get_daily_data(code, days=30)
            
            if df is None or df.empty:
                return False, "获取数据为空"
            
            saved_count = self.db.save_daily_data(df, code, source_name)
            return True, None
        except Exception as e:
            return False, str(e)

    def analyze_stock(self, code: str) -> Optional[AnalysisResult]:
        try:
            is_crypto_asset = self.is_crypto(code)
            stock_name = STOCK_NAME_MAP.get(code, code)
            extra_context = ""
            
            # --- 加密货币特有逻辑 ---
            if is_crypto_asset:
                logger.info(f"[{code}] 正在获取链上情绪数据...")
                extra_context = self.crypto_fetcher.get_onchain_sentiment()
                realtime_quote = None 
                chip_data = None
            else:
                # --- 原有股票逻辑 ---
                realtime_quote = self.akshare_fetcher.get_realtime_quote(code)
                if realtime_quote and realtime_quote.name:
                    stock_name = realtime_quote.name
                chip_data = self.akshare_fetcher.get_chip_distribution(code)

            # 获取技术面上下文
            context = self.db.get_analysis_context(code)
            if not context: return None
            
            # 趋势分析 (Crypto 同样适用 MA 趋势)
            import pandas as pd
            df = pd.DataFrame(context.get('raw_data', []))
            trend_result = self.trend_analyzer.analyze(df, code) if not df.empty else None

            # 增强上下文
            enhanced_context = self._enhance_context(context, realtime_quote, chip_data, trend_result, stock_name)
            
            # 搜索情报 (Crypto 也会搜索最新新闻)
            news_context = None
            if self.search_service.is_available:
                intel_results = self.search_service.search_comprehensive_intel(code, stock_name, max_searches=2)
                news_context = self.search_service.format_intel_report(intel_results, stock_name)

            # 调用 AI 分析 (传入 is_crypto 标记)
            result = self.analyzer.analyze(
                enhanced_context, 
                news_context=news_context, 
                extra_context=extra_context,
                is_crypto=is_crypto_asset  # 注意：需要在 analyzer.py 中适配此参数
            )
            return result
        except Exception as e:
            logger.error(f"[{code}] 分析异常: {e}")
            return None

    def _enhance_context(self, context, realtime_quote, chip_data, trend_result, stock_name):
        enhanced = context.copy()
        enhanced['stock_name'] = stock_name
        if realtime_quote:
            enhanced['realtime'] = {
                'price': realtime_quote.price,
                'volume_ratio': realtime_quote.volume_ratio,
                'turnover_rate': realtime_quote.turnover_rate
            }
        if chip_data:
            enhanced['chip'] = {'profit_ratio': chip_data.profit_ratio, 'chip_status': chip_data.get_chip_status(realtime_quote.price if realtime_quote else 0)}
        if trend_result:
            enhanced['trend_analysis'] = {
                'trend_status': trend_result.trend_status.value,
                'buy_signal': trend_result.buy_signal.value,
                'signal_score': trend_result.signal_score
            }
        return enhanced

    def process_single_stock(self, code: str, skip_analysis: bool = False, single_stock_notify: bool = False, report_type: ReportType = ReportType.SIMPLE) -> Optional[AnalysisResult]:
        logger.info(f"========== 开始处理 {code} ==========")
        success, error = self.fetch_and_save_stock_data(code)
        if skip_analysis: return None
        result = self.analyze_stock(code)
        if result and single_stock_notify and self.notifier.is_available():
            report_content = self.notifier.generate_single_stock_report(result)
            self.notifier.send(report_content)
        return result

    def run(self, stock_codes: Optional[List[str]] = None, dry_run: bool = False, send_notification: bool = True) -> List[AnalysisResult]:
        start_time = time.time()
        if stock_codes is None:
            self.config.refresh_stock_list()
            stock_codes = self.config.stock_list
        if not stock_codes: return []
        
        logger.info(f"开始执行任务，包含 {len(stock_codes)} 个标的")
        single_stock_notify = getattr(self.config, 'single_stock_notify', False)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {executor.submit(self.process_single_stock, code, dry_run, single_stock_notify and send_notification): code for code in stock_codes}
            for future in as_completed(future_to_code):
                try:
                    res = future.result()
                    if res: results.append(res)
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
        
        if results and send_notification and not dry_run:
            if not single_stock_notify:
                self._send_notifications(results)
        return results

    def _send_notifications(self, results: List[AnalysisResult]):
        try:
            report = self.notifier.generate_dashboard_report(results)
            self.notifier.save_report_to_file(report)
            if self.notifier.is_available():
                self.notifier.send(report)
        except Exception as e:
            logger.error(f"通知发送失败: {e}")

def run_full_analysis(config: Config, args: argparse.Namespace, stock_codes: Optional[List[str]] = None):
    try:
        pipeline = StockAnalysisPipeline(config=config, max_workers=args.workers)
        results = pipeline.run(stock_codes=stock_codes, dry_run=args.dry_run, send_notification=not args.no_notify)
        
        # 大盘复盘逻辑 (保持原样)
        market_report = ""
        if config.market_review_enabled and not args.no_market_review:
            market_report = run_market_review(pipeline.notifier, pipeline.analyzer, pipeline.search_service)

        # 飞书文档生成 (保持原样)
        try:
            feishu_doc = FeishuDocManager()
            if feishu_doc.is_configured() and (results or market_report):
                doc_title = f"{datetime.now().strftime('%Y-%m-%d %H:%M')} 投资复盘"
                full_content = ""
                if market_report: full_content += f"# 📈 大盘复盘\n\n{market_report}\n\n---\n\n"
                if results: full_content += f"# 🚀 决策仪表盘\n\n{pipeline.notifier.generate_dashboard_report(results)}"
                feishu_doc.create_daily_doc(doc_title, full_content)
        except Exception as e:
            logger.error(f"飞书生成失败: {e}")
    except Exception as e:
        logger.exception(f"执行失败: {e}")

def run_market_review(notifier, analyzer, search_service):
    try:
        market_analyzer = MarketAnalyzer(search_service=search_service, analyzer=analyzer)
        review_report = market_analyzer.run_daily_review()
        if review_report:
            notifier.send(f"🎯 大盘复盘\n\n{review_report}")
            return review_report
    except Exception as e:
        logger.error(f"大盘复盘失败: {e}")
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='股票 & 加密货币智能分析系统')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stocks', type=str)
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--schedule', action='store_true')
    parser.add_argument('--market-review', action='store_true')
    parser.add_argument('--no-market-review', action='store_true')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = get_config()
    setup_logging(debug=args.debug, log_dir=config.log_dir)
    
    stock_codes = [c.strip() for c in args.stocks.split(',')] if args.stocks else None
    
    if args.schedule or config.schedule_enabled:
        from scheduler import run_with_schedule
        run_with_schedule(task=lambda: run_full_analysis(config, args, stock_codes), schedule_time=config.schedule_time, run_immediately=True)
    else:
        run_full_analysis(config, args, stock_codes)
    return 0

if __name__ == "__main__":
    sys.exit(main())
