# fund_flow.py
"""
每日个股资金流采集脚本
数据源：akshare stock_individual_fund_flow（东方财富-个股资金流向）
"""
import os
import json
import time
import logging
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
DELAY_SEC = 3.0                               # 每只股票请求间隔（秒）
MAX_RETRY = 3                                 # 最大重试次数
DETAIL_FILE = "fundflow_detail.csv"           # 资金流历史明细（近100个交易日）
SUMMARY_FILE = "fundflow_summary.csv"         # 个股最新一日汇总指标
# =============================================================

COLUMN_MAP = {
    "日期": "date",
    "收盘价": "close",
    "涨跌幅": "pct_change",
    "主力净流入-净额": "main_net_inflow",
    "主力净流入-净占比": "main_net_inflow_rate",
    "超大单净流入-净额": "xlarge_net_inflow",
    "超大单净流入-净占比": "xlarge_net_inflow_rate",
    "大单净流入-净额": "large_net_inflow",
    "大单净流入-净占比": "large_net_inflow_rate",
    "中单净流入-净额": "mid_net_inflow",
    "中单净流入-净占比": "mid_net_inflow_rate",
    "小单净流入-净额": "small_net_inflow",
    "小单净流入-净占比": "small_net_inflow_rate",
}


def infer_market(code: str) -> str:
    """根据股票代码前缀推断交易所：沪/深/北"""
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8", "9")):
        return "bj"
    raise ValueError(f"无法识别股票代码 {code} 所属市场")


def fetch_fund_flow(code: str) -> pd.DataFrame:
    """抓取单只股票的资金流历史数据，失败时重试"""
    market = infer_market(code)
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is None or df.empty:
                raise ValueError("接口返回空数据")

            df = df.rename(columns=COLUMN_MAP)
            df["stock_code"] = code
            return df
        except Exception as err:
            logger.warning("[%s] 第 %d/%d 次抓取失败：%s", code, attempt, MAX_RETRY, err)
            if attempt < MAX_RETRY:
                time.sleep(DELAY_SEC * attempt)

    logger.error("[%s] 重试耗尽，跳过该股票", code)
    return pd.DataFrame()


def build_summary(df: pd.DataFrame) -> dict:
    """取最新一天的资金流数据作为汇总"""
    if df.empty:
        return {}

    latest = df.sort_values("date").iloc[-1]
    return {
        "stock_code": latest["stock_code"],
        "date": latest["date"],
        "close": latest["close"],
        "pct_change": latest["pct_change"],
        "main_net_inflow": latest["main_net_inflow"],
        "main_net_inflow_rate": latest["main_net_inflow_rate"],
        "xlarge_net_inflow": latest["xlarge_net_inflow"],
        "large_net_inflow": latest["large_net_inflow"],
        "mid_net_inflow": latest["mid_net_inflow"],
        "small_net_inflow": latest["small_net_inflow"],
    }


def main():
    detail_frames = []
    summary_rows = []

    for i, code in enumerate(STOCK_LIST):
        logger.info("正在抓取 %s (%d/%d)", code, i + 1, len(STOCK_LIST))
        flow_df = fetch_fund_flow(code)

        if not flow_df.empty:
            detail_frames.append(flow_df)
            summary_rows.append(build_summary(flow_df))
        else:
            logger.warning("[%s] 无数据，已跳过", code)

        if i < len(STOCK_LIST) - 1:
            time.sleep(DELAY_SEC)

    if not detail_frames:
        logger.error("本次未获取到任何资金流数据，退出")
        return

    detail_all = pd.concat(detail_frames, ignore_index=True)
    detail_all.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    logger.info("抓取完成 ✅  明细：%s  汇总：%s", DETAIL_FILE, SUMMARY_FILE)


if __name__ == "__main__":
    main()
