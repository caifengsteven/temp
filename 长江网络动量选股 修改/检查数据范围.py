# -*- coding: utf-8 -*-
"""
检查数据库中的数据时间范围
"""

from datayes_mysql_api import DataAPI
import pandas as pd

def check_data_range():
    """检查数据库中的数据时间范围"""
    
    print("=" * 80)
    print("检查数据库数据时间范围")
    print("=" * 80)
    
    # 1. 检查交易日历数据范围
    print("\n1. 检查交易日历数据...")
    try:
        # 获取最新的交易日历数据
        calendar_df = DataAPI.TradeCalGet(exchangeCD="XSHG", beginDate="2019-01-01", endDate="2024-12-31")
        if len(calendar_df) > 0:
            print(f"✓ 交易日历数据: {len(calendar_df)} 条记录")
            print(f"  日期范围: {calendar_df['calendarDate'].min()} 到 {calendar_df['calendarDate'].max()}")
            
            # 检查最新的周末交易日
            week_ends = calendar_df[calendar_df['isWeekEnd'] == 1]
            if len(week_ends) > 0:
                print(f"  周末交易日: {len(week_ends)} 个")
                print(f"  最新周末: {week_ends['calendarDate'].max()}")
        else:
            print("✗ 没有找到交易日历数据")
    except Exception as e:
        print(f"✗ 获取交易日历失败: {e}")
    
    # 2. 检查股票因子数据范围
    print("\n2. 检查股票因子数据...")
    try:
        # 尝试获取最新的因子数据
        test_dates = ['2024-01-05', '2023-01-06', '2022-01-07', '2021-01-08', '2020-01-10']
        
        latest_factor_date = None
        for test_date in test_dates:
            try:
                factor_data = DataAPI.MktStockFactorsOneDayGet(
                    tradeDate=test_date,
                    field=['secID', 'ticker', 'tradeDate', 'MassIndex', 'KDJ_J', 'RSI']
                )
                if len(factor_data) > 0:
                    latest_factor_date = test_date
                    print(f"✓ {test_date}: {len(factor_data)} 条因子记录")
                    break
                else:
                    print(f"  {test_date}: 无数据")
            except Exception as e:
                print(f"  {test_date}: 查询失败 - {e}")
        
        if latest_factor_date:
            print(f"✓ 最新因子数据日期: {latest_factor_date}")
        else:
            print("✗ 没有找到2020年以后的因子数据")
            
    except Exception as e:
        print(f"✗ 检查因子数据失败: {e}")
    
    # 3. 检查股票收益率数据范围
    print("\n3. 检查股票收益率数据...")
    try:
        # 尝试获取最新的收益率数据
        test_periods = [
            ('2024-01-01', '2024-01-31'),
            ('2023-01-01', '2023-01-31'), 
            ('2022-01-01', '2022-01-31'),
            ('2021-01-01', '2021-01-31'),
            ('2020-01-01', '2020-01-31')
        ]
        
        latest_return_period = None
        for start, end in test_periods:
            try:
                return_data = DataAPI.MktEquwAdjGet(
                    beginDate=start, 
                    endDate=end, 
                    field="secID,endDate,chgPct"
                )
                if len(return_data) > 0:
                    latest_return_period = (start, end)
                    print(f"✓ {start} 到 {end}: {len(return_data)} 条收益率记录")
                    print(f"  股票数量: {return_data['secID'].nunique()}")
                    print(f"  日期范围: {return_data['endDate'].min()} 到 {return_data['endDate'].max()}")
                    break
                else:
                    print(f"  {start} 到 {end}: 无数据")
            except Exception as e:
                print(f"  {start} 到 {end}: 查询失败 - {e}")
        
        if latest_return_period:
            print(f"✓ 最新收益率数据期间: {latest_return_period[0]} 到 {latest_return_period[1]}")
        else:
            print("✗ 没有找到2020年以后的收益率数据")
            
    except Exception as e:
        print(f"✗ 检查收益率数据失败: {e}")
    
    # 4. 总结
    print("\n" + "=" * 80)
    print("数据范围检查总结")
    print("=" * 80)
    
    if latest_factor_date and latest_return_period:
        print("✓ 数据库包含较新的数据，可以扩展回测期间")
        print(f"  建议扩展到: {latest_factor_date}")
    else:
        print("✗ 数据库可能只包含到2019年的数据")
        print("  建议检查数据源或更新数据库")

if __name__ == "__main__":
    check_data_range()
