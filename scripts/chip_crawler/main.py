# main.py
"""
每日筹码数据采集脚本
数据源：akshare stock_cyq_em（东方财富-筹码分布）
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

# stocks.json 与本脚本放在同一目录下，用脚本自身位置定位，
# 不依赖运行时的工作目录（避免在 GitHub Actions 里因 cwd 是仓库根目录而找不到文件）
DEFAULT_STOCKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.json")


def load_stock_list(path: str = DEFAULT_STOCKS_PATH) -> list:
    """从共享配置文件读取股票列表，三个采集脚本共用同一份"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stocks"]


# ===================== 配置区【按需修改】 =====================
STOCK_LIST = load_stock_list()                # 股票列表，来自 stocks.json
DELAY_SEC = 30.0                              # 每只股票请求间隔（秒）；stock_cyq_em 接口限流比其他接口更严重，单独拉更长
MAX_RETRY = 3                                 # 最大重试次数
RETRY_BASE_SEC = 20.0                         # 重试基础等待时间（秒），实际等待 = RETRY_BASE_SEC * 尝试次数
DETAIL_FILE = "chip_detail.csv"               # 筹码历史明细（每日一行）
SUMMARY_FILE = "chip_summary.csv"             # 个股最新一日汇总指标
# =============================================================

# ak.stock_cyq_em 实际返回的中文列名 -> 英文列名
COLUMN_MAP = {
    "日期": "date",
    "获利比例": "profit_rate",
    "平均成本": "avg_cost",
    "90成本-低": "cost90_low",
    "90成本-高": "cost90_high",
    "90集中度": "chip_concentration_90",
    "70成本-低": "cost70_low",
    "70成本-高": "cost70_high",
    "70集中度": "chip_concentration_70",
}


def fetch_chip_data(code: str) -> pd.DataFrame:
    """抓取单只股票的筹码分布历史数据，失败时重试"""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = ak.stock_cyq_em(symbol=code)
            if df is None or df.empty:
                raise ValueError("接口返回空数据")

            df = df.rename(columns=COLUMN_MAP)
            df["stock_code"] = code
            return df
        except Exception as err:
            logger.warning("[%s] 第 %d/%d 次抓取失败：%s", code, attempt, MAX_RETRY, err)
            if attempt < MAX_RETRY:
                time.sleep(RETRY_BASE_SEC * attempt)  # 递增等待，降低被限流概率

    logger.error("[%s] 重试耗尽，跳过该股票", code)
    return pd.DataFrame()


def build_summary(df: pd.DataFrame) -> dict:
    """从历史序列中取最新一天的数据作为汇总指标"""
    if df.empty:
        return {}

    latest = df.sort_values("date").iloc[-1]
    return {
        "stock_code": latest["stock_code"],
        "date": latest["date"],
        "avg_cost": latest["avg_cost"],
        "profit_rate": latest["profit_rate"],
        "chip_concentration_90": latest["chip_concentration_90"],
        "cost90_low": latest["cost90_low"],
        "cost90_high": latest["cost90_high"],
        "chip_concentration_70": latest["chip_concentration_70"],
        "cost70_low": latest["cost70_low"],
        "cost70_high": latest["cost70_high"],
    }


def main():
    detail_frames = []
    summary_rows = []

    for i, code in enumerate(STOCK_LIST):
        logger.info("正在抓取 %s (%d/%d)", code, i + 1, len(STOCK_LIST))
        chip_df = fetch_chip_data(code)

        if not chip_df.empty:
            detail_frames.append(chip_df)
            summary_rows.append(build_summary(chip_df))
        else:
            logger.warning("[%s] 无数据，已跳过", code)

        if i < len(STOCK_LIST) - 1:
            time.sleep(DELAY_SEC)

    if not detail_frames:
        logger.error("本次未获取到任何筹码数据，退出")
        return

    detail_all = pd.concat(detail_frames, ignore_index=True)
    detail_all.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    logger.info("抓取完成 ✅  明细：%s  汇总：%s", DETAIL_FILE, SUMMARY_FILE)


if __name__ == "__main__":
    main()
