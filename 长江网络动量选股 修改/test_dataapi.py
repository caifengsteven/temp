# -*- coding: utf-8 -*-
"""
测试DataAPI连接
"""

from datayes_mysql_api import DataAPI
import pandas as pd

def test_dataapi():
    """测试DataAPI功能"""
    
    print("测试DataAPI连接...")
    
    # 1. 测试交易日历
    print("\n1. 测试交易日历...")
    try:
        calendar_df = DataAPI.TradeCalGet(exchangeCD="XSHG", beginDate="2019-01-01", endDate="2019-01-10")
        print(f"获取到 {len(calendar_df)} 条交易日历数据")
        if len(calendar_df) > 0:
            print("样本数据:")
            print(calendar_df.head())
    except Exception as e:
        print(f"交易日历测试失败: {e}")
    
    # 2. 测试股票因子数据
    print("\n2. 测试股票因子数据...")
    try:
        factor_df = DataAPI.MktStockFactorsOneDayGet(
            tradeDate="2019-01-04",
            field=['secID', 'ticker', 'tradeDate', 'MassIndex', 'KDJ_J', 'RSI']
        )
        print(f"获取到 {len(factor_df)} 条因子数据")
        if len(factor_df) > 0:
            print("样本数据:")
            print(factor_df.head())
    except Exception as e:
        print(f"股票因子测试失败: {e}")
    
    # 3. 测试股票收益率数据
    print("\n3. 测试股票收益率数据...")
    try:
        returns_df = DataAPI.MktEquwAdjGet(
            beginDate="2019-01-01",
            endDate="2019-01-05",
            field="secID,endDate,chgPct"
        )
        print(f"获取到 {len(returns_df)} 条收益率数据")
        if len(returns_df) > 0:
            print("样本数据:")
            print(returns_df.head())
    except Exception as e:
        print(f"股票收益率测试失败: {e}")

if __name__ == "__main__":
    test_dataapi()
