# kline.py
"""
每日K线及技术指标采集脚本
数据源：akshare stock_zh_a_hist（东方财富-A股历史行情）
技术指标（MA5/MA10/MA20、MACD）在本地用 pandas 计算，无需额外依赖
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_STOCKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.json")


def load_stock_list(path: str = DEFAULT_STOCKS_PATH) -> list:
    """从共享配置文件读取股票列表，三个采集脚本共用同一份"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stocks"]


# ===================== 配置区【按需修改】 =====================
STOCK_LIST = load_stock_list()                # 股票列表，来自 stocks.json
DELAY_SEC = 15.0                              # 每只股票请求间隔（秒），拉长以避免触发东财限流
MAX_RETRY = 3                                 # 最大重试次数
RETRY_BASE_SEC = 10.0                         # 重试基础等待时间（秒），实际等待 = RETRY_BASE_SEC * 尝试次数
LOOKBACK_DAYS = 200                           # 拉取最近多少天的K线（保证MA20/MACD有足够历史预热）
ADJUST = "qfq"                                # 复权方式：qfq前复权 / hfq后复权 / "" 不复权
DETAIL_FILE = "kline_detail.csv"              # K线+技术指标历史明细
SUMMARY_FILE = "kline_summary.csv"            # 个股最新一日汇总指标
# =============================================================

COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
}


def get_date_range():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")


def fetch_kline(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓取单只股票的历史K线数据，失败时重试"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=ADJUST,
            )
            if df is None or df.empty:
                raise ValueError("接口返回空数据")

            df = df.rename(columns=COLUMN_MAP)
            df["stock_code"] = code
            return df
        except Exception as err:
            logger.warning("[%s] 第 %d/%d 次抓取失败：%s", code, attempt, MAX_RETRY, err)
            if attempt < MAX_RETRY:
                time.sleep(RETRY_BASE_SEC * attempt)

    logger.error("[%s] 重试耗尽，跳过该股票", code)
    return pd.DataFrame()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算均线和MACD，不依赖ta-lib等第三方库"""
    df = df.sort_values("date").reset_index(drop=True)

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2

    return df


def build_summary(df: pd.DataFrame) -> dict:
    """取最新一天的K线和技术指标作为汇总"""
    if df.empty:
        return {}

    latest = df.iloc[-1]
    return {
        "stock_code": latest["stock_code"],
        "date": latest["date"],
        "close": latest["close"],
        "pct_change": latest["pct_change"],
        "volume": latest["volume"],
        "turnover_rate": latest["turnover_rate"],
        "ma5": round(latest["ma5"], 2) if pd.notna(latest["ma5"]) else None,
        "ma10": round(latest["ma10"], 2) if pd.notna(latest["ma10"]) else None,
        "ma20": round(latest["ma20"], 2) if pd.notna(latest["ma20"]) else None,
        "dif": round(latest["dif"], 4),
        "dea": round(latest["dea"], 4),
        "macd": round(latest["macd"], 4),
    }


def main():
    start_date, end_date = get_date_range()
    logger.info("抓取区间：%s ~ %s", start_date, end_date)

    detail_frames = []
    summary_rows = []

    for i, code in enumerate(STOCK_LIST):
        logger.info("正在抓取 %s (%d/%d)", code, i + 1, len(STOCK_LIST))
        kline_df = fetch_kline(code, start_date, end_date)

        if not kline_df.empty:
            kline_df = add_technical_indicators(kline_df)
            detail_frames.append(kline_df)
            summary_rows.append(build_summary(kline_df))
        else:
            logger.warning("[%s] 无数据，已跳过", code)

        if i < len(STOCK_LIST) - 1:
            time.sleep(DELAY_SEC)

    if not detail_frames:
        logger.error("本次未获取到任何K线数据，退出")
        return

    detail_all = pd.concat(detail_frames, ignore_index=True)
    detail_all.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    logger.info("抓取完成 ✅  明细：%s  汇总：%s", DETAIL_FILE, SUMMARY_FILE)


if __name__ == "__main__":
    main()
