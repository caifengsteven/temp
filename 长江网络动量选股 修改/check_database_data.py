# -*- coding: utf-8 -*-
"""
检查数据库中的现有数据
"""

from data_manager import DataManager
import pandas as pd

def check_database_content():
    """检查数据库中的数据内容"""
    dm = DataManager()
    
    print("检查数据库内容...")
    print("=" * 50)
    
    # 检查交易日历
    try:
        calendar_df = dm.get_trade_calendar('2012-01-01', '2019-12-31')
        print(f"交易日历数据: {len(calendar_df)} 条记录")
        if len(calendar_df) > 0:
            print("交易日历样本数据:")
            print(calendar_df.head())
            print(f"日期范围: {calendar_df['calendar_date'].min()} 到 {calendar_df['calendar_date'].max()}")
        print()
    except Exception as e:
        print(f"获取交易日历失败: {e}")
    
    # 检查股票因子数据
    try:
        factors_df = dm.get_stock_factors('2012-01-01', '2019-12-31')
        print(f"股票因子数据: {len(factors_df)} 条记录")
        if len(factors_df) > 0:
            print("股票因子样本数据:")
            print(factors_df.head())
            print(f"股票数量: {factors_df['sec_id'].nunique()}")
            print(f"日期范围: {factors_df['trade_date'].min()} 到 {factors_df['trade_date'].max()}")
        print()
    except Exception as e:
        print(f"获取股票因子失败: {e}")
    
    # 检查股票收益率数据
    try:
        returns_df = dm.get_stock_returns('2012-01-01', '2019-12-31')
        print(f"股票收益率数据: {len(returns_df)} 条记录")
        if len(returns_df) > 0:
            print("股票收益率样本数据:")
            print(returns_df.head())
            print(f"股票数量: {returns_df['sec_id'].nunique()}")
            print(f"日期范围: {returns_df['end_date'].min()} 到 {returns_df['end_date'].max()}")
        print()
    except Exception as e:
        print(f"获取股票收益率失败: {e}")
    
    # 检查网络动量数据
    try:
        momentum_df = dm.get_network_momentum('2012-01-01', '2019-12-31')
        print(f"网络动量数据: {len(momentum_df)} 条记录")
        if len(momentum_df) > 0:
            print("网络动量样本数据:")
            print(momentum_df.head())
        print()
    except Exception as e:
        print(f"获取网络动量失败: {e}")
    
    dm.close()

if __name__ == "__main__":
    check_database_content()
