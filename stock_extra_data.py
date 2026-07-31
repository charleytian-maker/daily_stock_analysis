import time
import akshare as ak
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random

# 接口重试机制，解决Github Actions网络波动
@retry(stop=stop_after_attempt(3), wait=wait_random(min=0.3, max=0.8))
def get_stock_fund_flow(stock_code: str):
    """获取个股资金流向：主力净流入"""
    df = ak.stock_fund_flow_individual(symbol=stock_code)
    return {
        "main_net_inflow": df["主力净流入-净额"].iloc[0],
        "super_large_order": df["超大单-净额"].iloc[0],
        "large_order": df["大单-净额"].iloc[0]
    }

@retry(stop=stop_after_attempt(3), wait=wait_random(min=0.3, max=0.8))
def get_stock_chip_data(stock_code: str):
    """筹码基础数据：获利比例、平均成本、筹码区间"""
    df = ak.stock_chip_analysis(symbol=stock_code)
    return {
        "avg_cost": df["平均成本"].iloc[0],
        "profit_ratio": df["获利比例"].iloc[0],
        "chip_90_low": df["90%筹码下限"].iloc[0],
        "chip_90_high": df["90%筹码上限"].iloc[0]
    }

@retry(stop=stop_after_attempt(3), wait=wait_random(min=0.3, max=0.8))
def get_stock_financial_data(stock_code: str):
    """最新季度财务指标 ROE、毛利率、净利润增速"""
    df = ak.stock_financial_indicator_em(symbol=stock_code)
    latest = df.iloc[0]
    return {
        "roe": latest["净资产收益率"],
        "gross_margin": latest["销售毛利率"],
        "net_profit_growth": latest["净利润同比增长率"]
    }

def get_all_extra_info(stock_code: str):
    """统一入口：一次性获取三类数据"""
    try:
        fund = get_stock_fund_flow(stock_code)
        chip = get_stock_chip_data(stock_code)
        finance = get_stock_financial_data(stock_code)
        merge_data = {**fund, **chip, **finance}
        time.sleep(0.4)  # 请求限流，防止被接口封禁
        return merge_data
    except Exception as e:
        print(f"【{stock_code}】数据获取失败：{str(e)}")
        return None

# 选股筛选条件（阈值可自行修改）
def stock_qualify(extra_info) -> bool:
    if extra_info is None:
        return False
    # 1.主力资金净流入为正
    if extra_info["main_net_inflow"] <= 0:
        return False
    # 2.筹码获利比例大于30%
    if extra_info["profit_ratio"] < 30:
        return False
    # 3.净资产收益率ROE >5%
    if extra_info["roe"] < 5:
        return False
    # 4.毛利率大于10%
    if extra_info["gross_margin"] < 10:
        return False
    return True
