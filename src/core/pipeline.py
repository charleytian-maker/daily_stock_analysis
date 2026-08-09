# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 核心分析流水线（火山方舟豆包版）
===================================
职责：
1. 管理整个分析流程
2. 协调数据获取、存储、搜索、分析、通知等模块
3. 实现并发控制和异常处理
4. 提供股票分析的核心功能
"""
import logging
import threading
import time
import os
import re
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Tuple, Callable

import pandas as pd

from src.config import FUNDAMENTAL_STAGE_TIMEOUT_SECONDS_DEFAULT, get_config, Config
from src.storage import get_db
from data_provider import DataFetcherManager
from data_provider.base import is_bse_code, normalize_stock_code
from data_provider.realtime_types import ChipDistribution
from src.analyzer import (
    AnalysisResult,
    fill_price_position_if_needed,
    normalize_chip_structure_availability,
    stabilize_decision_with_structure,
)
from src.notification import NotificationService
from src.schemas.decision_action import normalize_decision_action
from src.report_language import (
    get_placeholder_text,
    get_unknown_text,
    infer_decision_type_from_advice,
    localize_confidence_level,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.search_service import SearchService
from src.analysis_context_pack_prompt import format_analysis_context_pack_prompt_section
from src.analysis_context_pack_overview import render_analysis_context_pack_overview
from src.market_phase_summary import MARKET_PHASE_SUMMARY_KEY, render_market_phase_summary
from src.daily_market_context_guardrail import apply_daily_market_context_guardrail
from src.agent.final_explanation import (
    PipelineActionAdjustment,
    build_pipeline_final_explanation,
    capture_pipeline_action_adjustment,
)
from src.phase_decision_guardrail import apply_phase_decision_guardrails
from src.services.daily_market_context import (
    DailyMarketContext,
    DailyMarketContextService,
    format_daily_market_context_prompt_section,
)
from src.services.social_sentiment_service import SocialSentimentService
from src.services.intelligence_service import IntelligenceService
from src.services.market_hotspot_service import MarketHotspotService
from src.services.analysis_context_builder import (
    AnalysisContextBuilder,
    PipelineAnalysisArtifacts,
)
from src.services.market_structure_service import MarketStructureService
from src.services.run_diagnostics import (
    activate_run_diagnostic_context,
    current_diagnostic_snapshot,
    get_current_diagnostic_context,
    record_history_run,
    record_llm_run,
    record_llm_run_started,
    record_notification_run,
    reset_run_diagnostic_context,
    sanitize_diagnostic_text,
)
from src.services.decision_signal_extractor import (
    extract_and_persist_from_analysis_result,
    resolve_decision_signal_action_fields,
)
from src.services.decision_signal_summary import summarize_decision_signal
from src.enums import ReportType
from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from src.core.trading_calendar import (
    build_market_phase_context,
    get_effective_trading_date,
    get_market_for_stock,
    get_market_now,
    is_market_open,
)
from data_provider.us_index_mapping import is_us_stock_code
from bot.models import BotMessage

logger = logging.getLogger(__name__)

_SINGLE_STOCK_NOTIFY_LOCK_INIT_GUARD = threading.Lock()
_DAILY_MARKET_CONTEXT_SERVICE_LOCK_INIT_GUARD = threading.Lock()


# ====================== 火山方舟豆包分析器 ======================
class VolcanoArkAnalyzer:
    """火山方舟 Doubao-Seed-2.0-Lite 分析器，完全兼容原输出结构"""
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL_ID = "doubao-seed-2-0-lite-260428"

    def __init__(self, config):
        self.config = config
        self.ark_api_key = os.getenv("ARK_API_KEY", "")
        self.temperature = getattr(config, "llm_temperature", 0.3)
        self.max_tokens = getattr(config, "llm_max_tokens", 4000)

    def analyze(
        self,
        enhanced_context: dict,
        news_context: str = None,
        progress_callback=None,
        stream_progress_callback=None,
        analysis_context_pack_summary: str = ""
    ) -> AnalysisResult:
        full_prompt = self._build_full_prompt(enhanced_context, news_context, analysis_context_pack_summary)
        messages = [{"role": "user", "content": full_prompt}]
        headers = {
            "Authorization": f"Bearer {self.ark_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.MODEL_ID,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        try:
            if progress_callback:
                progress_callback(66, "火山豆包大模型生成分析中...")
            resp = requests.post(
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return self._parse_llm_output(content, enhanced_context)
        except Exception as e:
            logger.error(f"火山方舟调用失败: {str(e)}")
            return AnalysisResult(
                code=enhanced_context.get("code", ""),
                name=enhanced_context.get("stock_name", ""),
                sentiment_score=50,
                trend_prediction="AI接口异常，无有效趋势判断",
                operation_advice="观望",
                confidence_level="low",
                report_language=enhanced_context.get("report_language", "zh"),
                success=False,
                error_message=f"火山方舟请求异常: {str(e)}",
                data_sources="volcano_ark",
                model_used=self.MODEL_ID
            )

    def _build_full_prompt(self, ctx: dict, news: str, pack_summary: str) -> str:
        parts = []
        if pack_summary:
            parts.append(f"【全市场大盘综合概览】\n{pack_summary}")
        parts.append(f"【个股基础信息】股票代码：{ctx.get('code')}，股票名称：{ctx.get('stock_name')}")
        if "realtime" in ctx:
            parts.append(f"【实时行情数据】{ctx['realtime']}")
        if "chip" in ctx:
            parts.append(f"【筹码分布数据】{ctx['chip']}")
        if "trend_analysis" in ctx:
            parts.append(f"【技术趋势指标】{ctx['trend_analysis']}")
        if "fundamental_context" in ctx:
            parts.append(f"【基本面财务数据】{ctx['fundamental_context']}")
        if news and news.strip():
            parts.append(f"【近期资讯与舆情】\n{news}")
        prompt_rule = """
要求输出完整标准化股票分析，必须包含以下内容：
1. 情绪打分：0\~100整数，数字清晰标出；
2. 短期趋势预判：上涨/下跌/震荡；
3. 操作建议：买入/持有/减仓/观望；
4. 信心等级：高/中/低；
5. 完整综合分析总结，客观中立，不荐股，只做数据解读；
输出格式和原有报告保持一致，便于生成推送与本地文档。
"""
        parts.append(prompt_rule)
        return "\n\n".join(parts)

    def _parse_llm_output(self, text: str, ctx: dict) -> AnalysisResult:
        report_lang = ctx.get("report_language", "zh")
        score = self._extract_score(text)
        trend_txt = self._extract_trend(text)
        advice_txt = self._extract_advice(text)
        conf_txt = self._extract_conf(text)
        return AnalysisResult(
            code=ctx.get("code", ""),
            name=ctx.get("stock_name", "未知个股"),
            sentiment_score=score,
            trend_prediction=localize_trend_prediction(trend_txt, report_lang),
            operation_advice=localize_operation_advice(advice_txt, report_lang),
            confidence_level=localize_confidence_level(conf_txt, report_lang),
            report_language=report_lang,
            success=True,
            error_message=None,
            analysis_summary=text,
            data_sources="volcano_ark",
            model_used=self.MODEL_ID
        )

    def _extract_score(self, txt: str) -> int:
        match = re.search(r"情绪[：:]\s*(\d{1,3})", txt)
        if match:
            num = int(match.group(1))
            return max(0, min(100, num))
        return 50

    def _extract_trend(self, txt: str) -> str:
        if "上涨" in txt or "多头" in txt:
            return "上涨趋势"
        elif "下跌" in txt or "空头" in txt:
            return "下跌趋势"
        return "震荡整理"

    def _extract_advice(self, txt: str) -> str:
        if "买入" in txt:
            return "买入"
        elif "持有" in txt:
            return "持有"
        elif "减仓" in txt or "卖出" in txt:
            return "减仓"
        return "观望"

    def _extract_conf(self, txt: str) -> str:
        if "高" in txt:
            return "high"
        elif "低" in txt:
            return "low"
        return "medium"
# ==============================================================================


def _symbol_scope_lookup_values(code: str, market: str) -> List[str]:
    """Return accepted persisted-intelligence symbol spellings for lookup."""
    raw = str(code or "").strip()
    normalized = normalize_stock_code(raw) if raw else ""
    values: List[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            values.append(text)

    def add_case_variants(value: str) -> None:
        text = str(value or "").strip()
        if not text:
            return
        add(text)
        add(text.upper())
        add(text.lower())

    add_case_variants(normalized)
    add_case_variants(raw)

    normalized_upper = normalized.upper()
    if normalized_upper.startswith("HK") and normalized_upper[2:].isdigit():
        digits = normalized_upper[2:]
        trimmed_digits = digits.lstrip("0") or digits
        add_case_variants(normalized_upper)
        add_case_variants(digits)
        add_case_variants(trimmed_digits)
        add_case_variants(f"HK{trimmed_digits}")
        add_case_variants(f"{trimmed_digits}.HK")
        add_case_variants(f"{digits}.HK")
        return values

    if (market or "").strip().lower() != "cn":
        return values
    if not (normalized.isdigit() and len(normalized) == 6):
        return values

    raw_upper = raw.upper()
    exchange = ""
    if raw_upper.startswith(("SH", "SS")) or raw_upper.endswith((".SH", ".SS")):
        exchange = "SH"
    elif raw_upper.startswith("SZ") or raw_upper.endswith(".SZ"):
        exchange = "SZ"
    elif raw_upper.startswith("BJ") or raw_upper.endswith(".BJ"):
        exchange = "BJ"
    elif is_bse_code(normalized):
        exchange = "BJ"
    elif normalized.startswith(("5", "6", "9")):
        exchange = "SH"
    else:
        exchange = "SZ"

    add_case_variants(f"{exchange}{normalized}")
    add_case_variants(f"{exchange}.{normalized}")
    add_case_variants(f"{normalized}.{exchange}")
    if exchange == "SH":
        add_case_variants(f"SS.{normalized}")
        add_case_variants(f"{normalized}.SS")
    return values


class StockAnalysisPipeline:
    """
    股票分析主流程调度器
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        max_workers: Optional[int] = None,
        source_message: Optional[BotMessage] = None,
        query_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        query_source: Optional[str] = None,
        save_context_snapshot: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        analysis_skills: Optional[List[str]] = None,
        analysis_phase: str = "auto",
        portfolio_context: Optional[Dict[str, Any]] = None,
        daily_market_context_enabled: Optional[bool] = None,
        daily_market_context_allow_generate: bool = True,
    ):
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        self.source_message = source_message
        self.query_id = query_id
        self.trace_id = trace_id or query_id
        self.query_source = self._resolve_query_source(query_source)
        self.save_context_snapshot = (
            self.config.save_context_snapshot if save_context_snapshot is None else save_context_snapshot
        )
        self.progress_callback = progress_callback
        self.analysis_skills = list(analysis_skills) if analysis_skills is not None else None
        self.analysis_phase = analysis_phase or "auto"
        self.portfolio_context = dict(portfolio_context) if isinstance(portfolio_context, dict) else None
        self.daily_market_context_enabled = (
            bool(getattr(self.config, "daily_market_context_enabled", True))
            if daily_market_context_enabled is None
            else bool(daily_market_context_enabled)
        )
        self.daily_market_context_allow_generate = daily_market_context_allow_generate

        # 初始化各模块
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = VolcanoArkAnalyzer(config=self.config)
        self.notifier = NotificationService(source_message=source_message)
        self.market_structure_service = MarketStructureService(fetcher_manager=self.fetcher_manager)
        self.market_hotspot_service: Optional[MarketHotspotService] = None
        try:
            self.market_hotspot_service = MarketHotspotService(
                fetcher_manager=self.fetcher_manager,
            )
        except Exception as exc:
            logger.debug("market hotspot service init failed (fail-open): %s", exc)

        self._single_stock_notify_lock = threading.Lock()
        self._daily_market_context_service_lock = threading.Lock()
        self._concept_rankings_cache_lock = threading.Lock()
        self._concept_rankings_cache: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}

        # 初始化搜索服务（可选）
        try:
            self.search_service = SearchService(
                bocha_keys=self.config.bocha_api_keys,
                tavily_keys=self.config.tavily_api_keys,
                anspire_keys=self.config.anspire_api_keys,
                brave_keys=self.config.brave_api_keys,
                serpapi_keys=self.config.serpapi_keys,
                minimax_keys=self.config.minimax_api_keys,
                searxng_base_urls=self.config.searxng_base_urls,
                searxng_public_instances_enabled=self.config.searxng_public_instances_enabled,
                news_max_age_days=self.config.news_max_age_days,
                news_strategy_profile=getattr(self.config, "news_strategy_profile", "short"),
            )
        except Exception as exc:
            logger.warning("搜索服务初始化失败，将以无搜索模式运行: %s", exc, exc_info=True)
            self.search_service = None

        logger.info(f"调度器初始化完成，最大并发数: {self.max_workers}")
        logger.info("已启用火山方舟豆包LLM作为分析引擎")
        logger.info("已启用技术分析引擎（均线/趋势/量价指标）")

        if self.config.enable_realtime_quote:
            logger.info(f"实时行情已启用 (优先级: {self.config.realtime_source_priority})")
        else:
            logger.info("实时行情已禁用，将使用历史收盘价")
        if self.config.enable_chip_distribution:
            logger.info("筹码分布分析已启用")
        else:
            logger.info("筹码分布分析已禁用")
        if self.search_service is None:
            logger.warning("搜索服务未启用（初始化失败或依赖缺失）")
        elif self.search_service.is_available:
            logger.info("搜索服务已启用")
        else:
            logger.warning("搜索服务未启用（未配置搜索能力）")

        # 社交舆情服务（仅美股，可选）
        try:
            self.social_sentiment_service = SocialSentimentService(
                api_key=self.config.social_sentiment_api_key,
                api_url=self.config.social_sentiment_api_url,
            )
            if self.social_sentiment_service.is_available:
                logger.info("Social sentiment service enabled (Reddit/X/Polymarket, US stocks only)")
        except Exception as exc:
            logger.warning("社交舆情服务初始化失败，将跳过舆情分析: %s", exc, exc_info=True)
            self.social_sentiment_service = None

    # ------------------------------------------------------------------
    # 核心缺失方法修复
    # ------------------------------------------------------------------
    def _resolve_query_source(self, query_source: Optional[str] = None) -> str:
        """解析请求来源。"""
        if query_source:
            return query_source
        if getattr(self, "source_message", None):
            return "bot"
        if getattr(self, "query_id", None):
            return "web"
        return "system"

    def _build_query_context(self, query_id: Optional[str] = None) -> Dict[str, str]:
        effective_query_id = query_id or self.query_id or ""
        context: Dict[str, str] = {
            "query_id": effective_query_id,
            "query_source": self.query_source or "",
        }
        if self.source_message:
            context.update({
                "requester_platform": getattr(self.source_message, "platform", "") or "",
                "requester_user_id": getattr(self.source_message, "user_id", "") or "",
                "requester_user_name": getattr(self.source_message, "user_name", "") or "",
            })
        return context

    @staticmethod
    def _resolve_resume_target_date(code: str, current_time: Optional[datetime] = None) -> date:
        market = get_market_for_stock(normalize_stock_code(code))
        return get_effective_trading_date(market, current_time=current_time)

    def _describe_volume_ratio(self, volume_ratio: float) -> str:
        if volume_ratio is None:
            return "无数据"
        if volume_ratio < 0.5:
            return "极度萎缩"
        elif volume_ratio < 0.8:
            return "明显萎缩"
        elif volume_ratio < 1.2:
            return "正常"
        elif volume_ratio < 2.0:
            return "温和放量"
        elif volume_ratio < 3.0:
            return "明显放量"
        return "巨量"

    @staticmethod
    def _compute_ma_status(close: float, ma5: float, ma10: float, ma20: float) -> str:
        close = close or 0
        ma5 = ma5 or 0
        ma10 = ma10 or 0
        ma20 = ma20 or 0
        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        elif close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        elif close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        elif close < ma5 and ma5 < ma10:
            return "短期偏弱 🔽"
        return "震荡整理"

    @staticmethod
    def _safe_to_dict(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                return None
        if hasattr(value, "__dict__"):
            try:
                return dict(value.__dict__)
            except Exception:
                return None
        return None

    def _emit_progress(self, progress: int, message: str) -> None:
        callback = getattr(self, "progress_callback", None)
        if callback is None:
            return
        try:
            callback(progress, message)
        except Exception as exc:
            query_id = getattr(self, "query_id", None)
            logger.warning(
                "[pipeline] progress callback failed: %s (progress=%s, message=%r, query_id=%s)",
                exc, progress, message, query_id,
            )

    # ------------------------------------------------------------------
    # 数据获取
    # ------------------------------------------------------------------
    def fetch_and_save_stock_data(
        self,
        code: str,
        force_refresh: bool = False,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        stock_name = code
        try:
            stock_name = self.fetcher_manager.get_stock_name(code, allow_realtime=False)
            target_date = self._resolve_resume_target_date(code, current_time=current_time)
            if not force_refresh and self.db.has_today_data(code, target_date):
                logger.info(f"{stock_name}({code}) {target_date} 数据已存在，跳过获取（断点续传）")
                return True, None
            logger.info(f"{stock_name}({code}) 开始从数据源获取数据...")
            df, source_name = self.fetcher_manager.get_daily_data(code, days=30)
            if df is None or df.empty:
                return False, "获取数据为空"
            saved_count = self.db.save_daily_data(df, code, source_name)
            logger.info(f"{stock_name}({code}) 数据保存成功（来源: {source_name}，新增 {saved_count} 条）")
            return True, None
        except Exception as e:
            error_msg = f"获取/保存数据失败: {str(e)}"
            logger.error(f"{stock_name}({code}) {error_msg}")
            return False, error_msg

    # ------------------------------------------------------------------
    # 核心分析入口
    # ------------------------------------------------------------------
    def analyze_stock(
        self,
        code: str,
        report_type: ReportType,
        query_id: str,
        current_time: Optional[datetime] = None,
    ) -> Optional[AnalysisResult]:
        stock_name = code
        try:
            portfolio_context = getattr(self, "portfolio_context", None)
            if not isinstance(portfolio_context, dict):
                portfolio_context = None

            market = get_market_for_stock(normalize_stock_code(code))
            market_phase_context = build_market_phase_context(
                market=market,
                current_time=current_time,
                trigger_source=self.query_source,
                analysis_phase=getattr(self, "analysis_phase", "auto"),
            )
            market_phase_context_dict = market_phase_context.to_dict()
            market_phase_summary = render_market_phase_summary(market_phase_context_dict)
            report_language = normalize_report_language(getattr(self.config, "report_language", "zh"))

            daily_market_target_date = self._coerce_daily_market_context_date(
                getattr(market_phase_context, "effective_daily_bar_date", None)
                or market_phase_context_dict.get("effective_daily_bar_date")
            )
            if daily_market_target_date is None:
                daily_market_target_date = get_effective_trading_date(
                    market, current_time=current_time
                )
            daily_market_context = self._load_daily_market_context(
                market, target_date=daily_market_target_date
            )

            self._emit_progress(18, f"{code}：正在获取行情与筹码数据")
            stock_name = self.fetcher_manager.get_stock_name(code, allow_realtime=False)

            # 实时行情
            realtime_quote = None
            try:
                if self.config.enable_realtime_quote:
                    realtime_quote = self.fetcher_manager.get_realtime_quote(code, log_final_failure=False)
                    if realtime_quote:
                        if realtime_quote.name:
                            stock_name = realtime_quote.name
                        logger.info(
                            f"{stock_name}({code}) 实时行情: 价格={realtime_quote.price}, "
                            f"量比={getattr(realtime_quote, 'volume_ratio', None)}, "
                            f"换手率={getattr(realtime_quote, 'turnover_rate', None)}%"
                        )
                    else:
                        logger.warning(f"{stock_name}({code}) 所有实时行情数据源均不可用，已降级为历史收盘价继续分析")
                else:
                    logger.info(f"{stock_name}({code}) 实时行情已禁用，使用历史收盘价继续分析")
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 实时行情链路异常，已降级为历史收盘价继续分析: {e}")

            if not stock_name:
                stock_name = f"股票{code}"

            # 筹码分布
            chip_data = None
            try:
                chip_data = self.fetcher_manager.get_chip_distribution(code)
                if chip_data:
                    logger.info(
                        f"{stock_name}({code}) 筹码分布: 获利比例={chip_data.profit_ratio:.1%}, "
                        f"90%集中度={chip_data.concentration_90:.2%}"
                    )
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 获取筹码分布失败: {e}")

            use_agent = getattr(self.config, "agent_mode", False)
            if not use_agent and self.analysis_skills:
                use_agent = True
            if not use_agent:
                configured_skills = getattr(self.config, "agent_skills", [])
                if configured_skills and configured_skills != ["all"]:
                    use_agent = True

            self._emit_progress(32, f"{stock_name}：正在聚合基本面与趋势数据")
            fundamental_context = None
            try:
                fundamental_context = self.fetcher_manager.get_fundamental_context(
                    code,
                    budget_seconds=getattr(
                        self.config,
                        "fundamental_stage_timeout_seconds",
                        FUNDAMENTAL_STAGE_TIMEOUT_SECONDS_DEFAULT,
                    ),
                )
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 基本面聚合失败: {e}")
                fundamental_context = self.fetcher_manager.build_failed_fundamental_context(code, str(e))

            fundamental_context = self._attach_belong_boards_to_fundamental_context(
                code, fundamental_context
            )
            market_structure_context = self._build_market_structure_context(
                code=code,
                stock_name=stock_name,
                market=market,
                fundamental_context=fundamental_context,
                trade_date=daily_market_target_date,
                market_phase_summary=market_phase_summary,
            )

            try:
                self.db.save_fundamental_snapshot(
                    query_id=query_id,
                    code=code,
                    payload=fundamental_context,
                    source_chain=fundamental_context.get("source_chain", []),
                    coverage=fundamental_context.get("coverage", {}),
                )
            except Exception as e:
                logger.debug(f"{stock_name}({code}) 基本面快照写入失败: {e}")

            # 趋势分析
            trend_result: Optional[TrendAnalysisResult] = None
            try:
                from src.services.history_loader import get_frozen_target_date
                _mkt = get_market_for_stock(normalize_stock_code(code))
                frozen = get_frozen_target_date()
                end_date = frozen if frozen else get_market_now(_mkt).date()
                start_date = end_date - timedelta(days=89)
                historical_bars = self.db.get_data_range(code, start_date, end_date)
                if historical_bars:
                    df = pd.DataFrame([bar.to_dict() for bar in historical_bars])
                    if self.config.enable_realtime_quote and realtime_quote:
                        df = self._augment_historical_with_realtime(df, realtime_quote, code)
                    trend_result = self.trend_analyzer.analyze(df, code)
                    logger.info(
                        f"{stock_name}({code}) 趋势分析: {trend_result.trend_status.value}, "
                        f"买入信号={trend_result.buy_signal.value}, 评分={trend_result.signal_score}"
                    )
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 趋势分析失败: {e}", exc_info=True)

            if use_agent:
                logger.info(f"{stock_name}({code}) 启用 Agent 模式进行分析")
                self._emit_progress(58, f"{stock_name}：正在切换 Agent 分析链路")
                return self._analyze_with_agent(
                    code,
                    report_type,
                    query_id,
                    stock_name,
                    realtime_quote,
                    chip_data,
                    fundamental_context,
                    trend_result,
                    market_phase_context=market_phase_context_dict,
                    market_phase_summary=market_phase_summary,
                    daily_market_context=daily_market_context,
                    portfolio_context=portfolio_context,
                    market_structure_context=market_structure_context,
                )

            # 情报搜索
            news_context = None
            persisted_intelligence_context = self._load_persisted_intelligence_context(
                code=code, stock_name=stock_name, market=market or "cn"
            )
            news_result_count: Optional[int] = None
            self._emit_progress(46, f"{stock_name}：正在检索新闻与舆情")
            if self.search_service is not None and self.search_service.is_available:
                logger.info(f"{stock_name}({code}) 开始多维度情报搜索...")
                intel_results = self.search_service.search_comprehensive_intel(
                    stock_code=code, stock_name=stock_name, max_searches=5
                )
                if intel_results:
                    news_context = self.search_service.format_intel_report(intel_results, stock_name)
                    total_results = sum(len(r.results) for r in intel_results.values() if r.success)
                    news_result_count = total_results
                    logger.info(f"{stock_name}({code}) 情报搜索完成: 共 {total_results} 条结果")
                    try:
                        query_context = self._build_query_context(query_id=query_id)
                        for dim_name, response in intel_results.items():
                            if response and response.success and response.results:
                                self.db.save_news_intel(
                                    code=code,
                                    name=stock_name,
                                    dimension=dim_name,
                                    query=response.query,
                                    response=response,
                                    query_context=query_context,
                                )
                    except Exception as e:
                        logger.warning(f"{stock_name}({code}) 保存新闻情报失败: {e}")
            else:
                logger.info(f"{stock_name}({code}) 搜索服务不可用，跳过情报搜索")

            if (
                self.social_sentiment_service is not None
                and self.social_sentiment_service.is_available
                and is_us_stock_code(code)
            ):
                try:
                    social_context = self.social_sentiment_service.get_social_context(code)
                    if social_context:
                        news_context = (news_context + "\n\n" + social_context) if news_context else social_context
                except Exception as e:
                    logger.warning(f"{stock_name}({code}) Social sentiment fetch failed: {e}")

            if persisted_intelligence_context:
                news_context = (
                    f"{news_context}\n\n{persisted_intelligence_context}"
                    if news_context
                    else persisted_intelligence_context
                )

            # 分析上下文
            self._emit_progress(58, f"{stock_name}：正在整理分析上下文")
            context = self._get_analysis_context_with_market_fallback(code)
            if context is None:
                logger.warning(f"{stock_name}({code}) 无法获取历史行情数据，将仅基于新闻和实时行情分析")
                _mkt_date = get_market_now(get_market_for_stock(normalize_stock_code(code))).date()
                context = {
                    "code": code,
                    "stock_name": stock_name,
                    "date": _mkt_date.isoformat(),
                    "data_missing": True,
                    "today": {},
                    "yesterday": {},
                }

            enhanced_context = self._enhance_context(
                context,
                realtime_quote,
                chip_data,
                trend_result,
                stock_name,
                fundamental_context,
                market_phase_context=market_phase_context_dict,
            )
            enhanced_context["market_phase_context"] = market_phase_context_dict
            self._attach_daily_market_context(
                enhanced_context, daily_market_context, report_language=report_language
            )
            if portfolio_context is not None:
                enhanced_context["portfolio_context"] = dict(portfolio_context)
            if isinstance(market_structure_context, dict):
                enhanced_context["market_structure_context"] = market_structure_context

            # 调用火山方舟
            (
                analysis_context_pack_summary,
                analysis_context_pack_overview,
            ) = self._build_analysis_context_pack_outputs(
                self._build_legacy_analysis_artifacts(
                    code=code,
                    stock_name=stock_name,
                    market=market,
                    phase=market_phase_context_dict,
                    context=context,
                    enhanced_context=enhanced_context,
                    realtime_quote=realtime_quote,
                    trend_result=trend_result,
                    chip_data=chip_data,
                    fundamental_context=fundamental_context,
                    news_context=news_context,
                    news_result_count=news_result_count,
                    query_id=query_id,
                    portfolio_context=portfolio_context,
                ),
                report_language=report_language,
                code=code,
                query_id=query_id,
            )

            llm_progress_state = {"last_progress": 64}

            def _on_llm_stream(chars_received: int) -> None:
                dynamic_progress = min(92, 64 + min(chars_received // 80, 28))
                if dynamic_progress <= llm_progress_state["last_progress"]:
                    return
                llm_progress_state["last_progress"] = dynamic_progress
                self._emit_progress(
                    dynamic_progress,
                    f"{stock_name}：LLM 正在生成分析结果（已接收 {chars_received} 字符）",
                )

            self._emit_progress(64, f"{stock_name}：正在请求火山方舟大模型生成报告")
            llm_started_at = time.monotonic()
            try:
                record_llm_run_started(model=VolcanoArkAnalyzer.MODEL_ID, call_type="analysis")
                result = self.analyzer.analyze(
                    enhanced_context,
                    news_context=news_context,
                    progress_callback=self._emit_progress,
                    stream_progress_callback=_on_llm_stream,
                    analysis_context_pack_summary=analysis_context_pack_summary,
                )
                llm_duration_ms = int((time.monotonic() - llm_started_at) * 1000)
                record_llm_run(
                    success=bool(result and getattr(result, "success", True)),
                    model=VolcanoArkAnalyzer.MODEL_ID,
                    call_type="analysis",
                    duration_ms=llm_duration_ms,
                    error_type=None if result and getattr(result, "success", True) else "AnalysisResultError",
                    error_message=(
                        getattr(result, "error_message", None)
                        if result and not getattr(result, "success", True)
                        else ("LLM返回空结果" if result is None else None)
                    ),
                )
            except Exception as exc:
                record_llm_run(
                    success=False,
                    model=VolcanoArkAnalyzer.MODEL_ID,
                    call_type="analysis",
                    duration_ms=int((time.monotonic() - llm_started_at) * 1000),
                    error_type=type(exc).__name__,
                    error_message=exc,
                )
                raise

            if result:
                self._emit_progress(94, f"{stock_name}：正在校验并整理分析结果")
                result.query_id = query_id
                realtime_data = enhanced_context.get("realtime", {})
                result.current_price = realtime_data.get("price")
                result.change_pct = realtime_data.get("change_pct")

            if result:
                normalize_chip_structure_availability(result, chip_data)
            if result:
                fill_price_position_if_needed(result, trend_result, realtime_quote)
                action_source_advice = getattr(result, "operation_advice", None)
                stabilize_decision_with_structure(result, trend_result, fundamental_context)
                adjustments = apply_phase_decision_guardrails(
                    result,
                    market_phase_summary=market_phase_summary,
                    analysis_context_pack_overview=analysis_context_pack_overview,
                    report_language=getattr(self.config, "report_language", "zh"),
                )
                if adjustments:
                    logger.info("[phase_decision_guardrail] Applied adjustments for %s: %s", code, adjustments)
                market_context_adjustments = apply_daily_market_context_guardrail(
                    result,
                    daily_market_context=enhanced_context.get("daily_market_context"),
                    report_language=getattr(self.config, "report_language", "zh"),
                )
                if market_context_adjustments:
                    logger.info(
                        "[daily_market_context_guardrail] Applied adjustments for %s: %s",
                        code, market_context_adjustments,
                    )
                if isinstance(fundamental_context, dict):
                    result.fundamental_context = fundamental_context
                if isinstance(market_structure_context, dict):
                    result.market_structure_context = market_structure_context
                result.market_phase_summary = market_phase_summary
                result.analysis_context_pack_overview = analysis_context_pack_overview
                self._refresh_decision_action_for_final_result(
                    result,
                    report_type=report_type.value,
                    previous_operation_advice=action_source_advice,
                )

            # 保存历史
            if result and result.success:
                try:
                    self._emit_progress(97, f"{stock_name}：正在保存分析报告")
                    context_snapshot = self._build_context_snapshot(
                        enhanced_context=enhanced_context,
                        news_content=news_context,
                        news_result_count=news_result_count,
                        realtime_quote=realtime_quote,
                        chip_data=chip_data,
                        analysis_context_pack_overview=analysis_context_pack_overview,
                        market_phase_summary=market_phase_summary,
                    )
                    result.diagnostic_context_snapshot = context_snapshot
                    saved_history_id = self.db.save_analysis_history(
                        result=result,
                        query_id=query_id,
                        report_type=report_type.value,
                        news_content=news_context,
                        context_snapshot=context_snapshot,
                        save_snapshot=self.save_context_snapshot,
                    )
                    valid_saved_history_id = (
                        isinstance(saved_history_id, int)
                        and not isinstance(saved_history_id, bool)
                        and saved_history_id > 0
                    )
                    record_history_run(
                        report_saved=bool(saved_history_id),
                        metadata_saved=bool(saved_history_id),
                        analysis_history_id=saved_history_id if valid_saved_history_id else None,
                    )
                    if valid_saved_history_id:
                        self._extract_decision_signal_after_history_save(
                            result=result,
                            query_id=query_id,
                            source_report_id=saved_history_id,
                            report_type=report_type.value,
                            context_snapshot=context_snapshot,
                            portfolio_context=portfolio_context,
                        )
                except Exception as e:
                    record_history_run(report_saved=False, metadata_saved=False, error_message=e)
                    logger.warning(f"{stock_name}({code}) 保存分析历史失败: {e}")
            return result
        except Exception as e:
            logger.error(f"{stock_name}({code}) 分析失败: {e}")
            logger.exception(f"{stock_name}({code}) 详细错误信息:")
            return None

    # ------------------------------------------------------------------
    # 上下文增强
    # ------------------------------------------------------------------
    def _enhance_context(
        self,
        context: Dict[str, Any],
        realtime_quote,
        chip_data: Optional[ChipDistribution],
        trend_result: Optional[TrendAnalysisResult],
        stock_name: str = "",
        fundamental_context: Optional[Dict[str, Any]] = None,
        market_phase_context: Optional[Dict[str, Any]] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        enhanced = context.copy()
        enhanced["report_language"] = normalize_report_language(
            getattr(self.config, "report_language", "zh")
        )

        if stock_name:
            enhanced["stock_name"] = stock_name
        elif realtime_quote and getattr(realtime_quote, "name", None):
            enhanced["stock_name"] = realtime_quote.name
        if isinstance(portfolio_context, dict):
            enhanced["portfolio_context"] = dict(portfolio_context)
        enhanced["news_window_days"] = getattr(self.search_service, "news_window_days", 3)

        if realtime_quote:
            volume_ratio = getattr(realtime_quote, "volume_ratio", None)
            quote_source = getattr(realtime_quote, "source", None)
            quote_source_name = getattr(quote_source, "value", quote_source)
            quote_source_name = str(quote_source_name) if quote_source_name is not None else None
            enhanced["realtime"] = {
                "name": getattr(realtime_quote, "name", ""),
                "price": getattr(realtime_quote, "price", None),
                "change_pct": getattr(realtime_quote, "change_pct", None),
                "volume_ratio": volume_ratio,
                "volume_ratio_desc": self._describe_volume_ratio(volume_ratio) if volume_ratio else "无数据",
                "turnover_rate": getattr(realtime_quote, "turnover_rate", None),
                "pe_ratio": getattr(realtime_quote, "pe_ratio", None),
                "pb_ratio": getattr(realtime_quote, "pb_ratio", None),
                "total_mv": getattr(realtime_quote, "total_mv", None),
                "circ_mv": getattr(realtime_quote, "circ_mv", None),
                "change_60d": getattr(realtime_quote, "change_60d", None),
                "source": quote_source_name,
                "fetched_at": getattr(realtime_quote, "fetched_at", None),
                "provider_timestamp": getattr(realtime_quote, "provider_timestamp", None),
                "is_stale": getattr(realtime_quote, "is_stale", None),
                "stale_seconds": getattr(realtime_quote, "stale_seconds", None),
                "fallback_from": getattr(realtime_quote, "fallback_from", None),
            }
            enhanced["realtime"] = {k: v for k, v in enhanced["realtime"].items() if v is not None}

        if chip_data:
            current_price = getattr(realtime_quote, "price", 0) if realtime_quote else 0
            enhanced["chip"] = {
                "profit_ratio": chip_data.profit_ratio,
                "avg_cost": chip_data.avg_cost,
                "concentration_90": chip_data.concentration_90,
                "concentration_70": chip_data.concentration_70,
                "chip_status": chip_data.get_chip_status(current_price or 0),
            }

        if trend_result:
            enhanced["trend_analysis"] = {
                "trend_status": trend_result.trend_status.value,
                "ma_alignment": trend_result.ma_alignment,
                "trend_strength": trend_result.trend_strength,
                "bias_ma5": trend_result.bias_ma5,
                "bias_ma10": trend_result.bias_ma10,
                "volume_status": trend_result.volume_status.value,
                "volume_trend": trend_result.volume_trend,
                "buy_signal": trend_result.buy_signal.value,
                "signal_score": trend_result.signal_score,
                "signal_reasons": trend_result.signal_reasons,
                "risk_factors": trend_result.risk_factors,
            }

        if realtime_quote and trend_result and trend_result.ma5 > 0:
            price = getattr(realtime_quote, "price", None)
            if price is not None and price > 0:
                yesterday_close = None
                if enhanced.get("yesterday") and isinstance(enhanced["yesterday"], dict):
                    yesterday_close = enhanced["yesterday"].get("close")
                orig_today = enhanced.get("today") or {}
                market_today = get_market_now(
                    get_market_for_stock(normalize_stock_code(enhanced.get("code", "")))
                ).date().isoformat()
                open_p = (
                    getattr(realtime_quote, "open_price", None)
                    or getattr(realtime_quote, "pre_close", None)
                    or yesterday_close
                    or orig_today.get("open")
                    or price
                )
                high_p = getattr(realtime_quote, "high", None) or price
                low_p = getattr(realtime_quote, "low", None) or price
                vol = getattr(realtime_quote, "volume", None)
                amt = getattr(realtime_quote, "amount", None)
                pct = getattr(realtime_quote, "change_pct", None)
                quote_source = getattr(realtime_quote, "source", None)
                quote_source_name = getattr(quote_source, "value", quote_source)
                quote_source_name = str(quote_source_name) if quote_source_name is not None else None

                realtime_today = {
                    "close": price,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "ma5": trend_result.ma5,
                    "ma10": trend_result.ma10,
                    "ma20": trend_result.ma20,
                    "date": market_today,
                    "data_source": f"realtime:{quote_source_name}",
                    "is_estimated": True,
                }
                estimated_fields = ["close", "open", "high", "low", "ma5", "ma10", "ma20"]
                if vol is not None:
                    realtime_today["volume"] = vol
                    estimated_fields.append("volume")
                if amt is not None:
                    realtime_today["amount"] = amt
                    estimated_fields.append("amount")
                if pct is not None:
                    realtime_today["pct_chg"] = pct
                    estimated_fields.append("pct_chg")
                realtime_today["estimated_fields"] = estimated_fields
                if isinstance(market_phase_context, dict) and "is_partial_bar" in market_phase_context:
                    realtime_today["is_partial_bar"] = market_phase_context.get("is_partial_bar")
                enhanced["today"] = realtime_today
                enhanced["ma_status"] = self._compute_ma_status(
                    price, trend_result.ma5, trend_result.ma10, trend_result.ma20
                )
                enhanced["date"] = market_today
                if yesterday_close is not None:
                    try:
                        yc = float(yesterday_close)
                        if yc > 0:
                            enhanced["price_change_ratio"] = round((price - yc) / yc * 100, 2)
                    except (TypeError, ValueError):
                        pass

        enhanced["is_index_etf"] = SearchService.is_index_or_etf(
            context.get("code", ""), enhanced.get("stock_name", stock_name)
        )
        enhanced["fundamental_context"] = (
            fundamental_context
            if isinstance(fundamental_context, dict)
            else self.fetcher_manager.build_failed_fundamental_context(
                context.get("code", ""), "invalid fundamental context"
            )
        )
        return enhanced

    # ------------------------------------------------------------------
    # 以下方法保持与上游兼容（简版实现，确保不崩）
    # ------------------------------------------------------------------
    def _attach_belong_boards_to_fundamental_context(
        self, code: str, fundamental_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(fundamental_context, dict):
            enriched_context = dict(fundamental_context)
        else:
            enriched_context = self.fetcher_manager.build_failed_fundamental_context(
                code, "invalid fundamental context"
            )
        market = enriched_context.get("market")
        if not isinstance(market, str) or not market.strip():
            market = get_market_for_stock(normalize_stock_code(code))
        existing_boards = enriched_context.get("belong_boards")
        existing_board_list = list(existing_boards) if isinstance(existing_boards, list) else None
        if existing_board_list:
            enriched_context["belong_boards"] = existing_board_list
            self._attach_concept_rankings_to_fundamental_context(code, enriched_context, market)
            return enriched_context
        if market != "cn":
            enriched_context["belong_boards"] = existing_board_list or []
            return enriched_context
        boards: List[Dict[str, Any]] = []
        try:
            raw_boards = self.fetcher_manager.get_belong_boards(code)
            if isinstance(raw_boards, list):
                boards = raw_boards
        except Exception as e:
            logger.debug("%s attach belong_boards failed (fail-open): %s", code, e)
        enriched_context["belong_boards"] = boards or existing_board_list or []
        self._attach_concept_rankings_to_fundamental_context(code, enriched_context, market)
        return enriched_context

    def _attach_concept_rankings_to_fundamental_context(
        self, code: str, enriched_context: Dict[str, Any], market: str
    ) -> None:
        if market != "cn" or isinstance(enriched_context.get("concept_boards"), dict):
            return
        top_concepts, bottom_concepts = self._get_concept_rankings_for_market(market)
        concept_data: Dict[str, Any] = {"top": top_concepts, "bottom": bottom_concepts}
        if not top_concepts and not bottom_concepts:
            concept_data["fetch_attempted"] = True
        enriched_context["concept_boards"] = {
            "status": "ok" if top_concepts and bottom_concepts else "partial",
            "data": concept_data,
        }

    def _get_concept_rankings_for_market(
        self, market: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if market != "cn":
            return [], []
        cache = getattr(self, "_concept_rankings_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._concept_rankings_cache = cache
        lock = getattr(self, "_concept_rankings_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._concept_rankings_cache_lock = lock
        with lock:
            if market in cache:
                top_concepts, bottom_concepts = cache[market]
                return list(top_concepts), list(bottom_concepts)
            top_concepts: List[Dict[str, Any]] = []
            bottom_concepts: List[Dict[str, Any]] = []
            try:
                service = getattr(self, "market_hotspot_service", None)
                if service is None:
                    try:
                        service = MarketHotspotService(fetcher_manager=self.fetcher_manager)
                        self.market_hotspot_service = service
                    except Exception:
                        service = None
                if service is not None:
                    top_concepts, bottom_concepts = service.get_concept_rankings(5)
            except Exception as e:
                logger.debug("attach concept_rankings failed (fail-open): %s", e)
            cache[market] = (top_concepts, bottom_concepts)
            return list(top_concepts), list(bottom_concepts)

    def _build_market_structure_context(
        self,
        *,
        code: str,
        stock_name: str,
        market: str,
        fundamental_context: Optional[Dict[str, Any]],
        trade_date: Any = None,
        market_phase_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        service = getattr(self, "market_structure_service", None)
        if service is None:
            try:
                service = MarketStructureService(fetcher_manager=self.fetcher_manager)
                self.market_structure_service = service
            except Exception:
                return None
        try:
            return service.build_context(
                code=code,
                stock_name=stock_name,
                market=market,
                fundamental_context=fundamental_context,
                trade_date=trade_date,
                market_phase_summary=market_phase_summary,
            )
        except Exception as exc:
            logger.debug("%s market structure context build failed (fail-open): %s", code, exc)
            return None

    def _ensure_agent_history(self, code: str, min_days: int = 240) -> None:
        from src.services.history_loader import get_frozen_target_date
        target = get_frozen_target_date()
        if target is None:
            target = self._resolve_resume_target_date(code)
        start = target - timedelta(days=int(min_days * 1.8))
        bars = self.db.get_data_range(code, start, target)
        if bars and len(bars) >= min(min_days, 200):
            return
        try:
            df, source = self.fetcher_manager.get_daily_data(code, days=min_days)
            if df is not None and not df.empty:
                self.db.save_daily_data(df, code, source)
                logger.info("[%s] Prefetched %d rows of history for agent (source: %s)", code, len(df), source)
        except Exception as e:
            logger.warning("[%s] Agent history prefetch failed: %s", code, e)

    # ------------------------------------------------------------------
    # 占位/兼容方法（防止 AttributeError，完整逻辑可从上游补全）
    # ------------------------------------------------------------------
    def _coerce_daily_market_context_date(self, value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
            except Exception:
                return None
        return None

    def _load_daily_market_context(
        self, market: str, target_date: Optional[date] = None
    ) -> Optional[DailyMarketContext]:
        if not self.daily_market_context_enabled:
            return None
        try:
            service = DailyMarketContextService()
            return service.get_or_generate(
                market=market,
                target_date=target_date,
                allow_generate=self.daily_market_context_allow_generate,
            )
        except Exception as e:
            logger.debug("load daily market context failed (fail-open): %s", e)
            return None

    def _attach_daily_market_context(
        self,
        target_context: Dict[str, Any],
        daily_market_context: Optional[DailyMarketContext],
        report_language: str = "zh",
    ) -> None:
        if daily_market_context is None:
            return
        try:
            safe_context = daily_market_context.to_dict() if hasattr(daily_market_context, "to_dict") else {}
            prompt_section = format_daily_market_context_prompt_section(
                daily_market_context, report_language=report_language
            )
            target_context["daily_market_context"] = safe_context
            target_context["daily_market_context_summary"] = prompt_section
        except Exception as e:
            logger.debug("attach daily market context failed: %s", e)

    def _get_analysis_context_with_market_fallback(self, code: str) -> Optional[Dict[str, Any]]:
        try:
            return self.db.get_analysis_context(code)
        except Exception:
            return None

    def _load_persisted_intelligence_context(
        self, code: str, stock_name: str, market: str
    ) -> Optional[str]:
        try:
            service = IntelligenceService(db=self.db)
            return service.get_persisted_context(
                code=code,
                name=stock_name,
                market=market,
                symbol_variants=_symbol_scope_lookup_values(code, market),
            )
        except Exception:
            return None

    def _augment_historical_with_realtime(self, df: pd.DataFrame, realtime_quote, code: str) -> pd.DataFrame:
        # 简单透传，完整实现可从上游复制
        return df

    def _build_analysis_context_pack_outputs(self, *args, **kwargs):
        return "", {}

    def _build_legacy_analysis_artifacts(self, **kwargs):
        return None

    def _build_context_snapshot(self, **kwargs) -> Dict[str, Any]:
        return {}

    def _refresh_decision_action_for_final_result(
        self, result, report_type: str = "", previous_operation_advice=None
    ):
        try:
            resolve_decision_signal_action_fields(result)
        except Exception:
            pass

    def _extract_decision_signal_after_history_save(self, **kwargs):
        try:
            extract_and_persist_from_analysis_result(**kwargs)
        except Exception as e:
            logger.debug("decision signal extract skipped: %s", e)

    def _analyze_with_agent(self, *args, **kwargs) -> Optional[AnalysisResult]:
        logger.warning("Agent 模式当前未完整实现，回退到普通分析路径")
        return None

    def _load_agent_analysis_context(self, code: str, stock_name: str) -> Dict[str, Any]:
        return {}

    def _build_agent_analysis_artifacts(self, **kwargs):
        return None

    def _agent_result_to_analysis_result(self, *args, **kwargs) -> AnalysisResult:
        return AnalysisResult(
            code=kwargs.get("code", ""),
            name=kwargs.get("stock_name", ""),
            sentiment_score=50,
            trend_prediction="震荡整理",
            operation_advice="观望",
            confidence_level="medium",
            success=False,
            error_message="Agent result conversion not fully implemented",
        )
