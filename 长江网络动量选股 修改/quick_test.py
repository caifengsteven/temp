# -*- coding: utf-8 -*-
"""
快速测试最终步骤
"""

from datayes_mysql_api import DataAPI
import pandas as pd
import numpy as np
from scipy import stats

def quick_test():
    """快速测试策略的最终步骤"""
    
    print("快速测试网络动量策略最终步骤...")
    
    # 使用很小的数据集进行测试
    start_date = "2019-01-01"
    end_date = "2019-01-31"
    
    print(f"测试期间: {start_date} 到 {end_date}")
    
    # 1. 获取交易日历
    calendar_df = DataAPI.TradeCalGet(exchangeCD="XSHG", beginDate=start_date, endDate=end_date)
    week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
    print(f"周末日期: {week_end_list}")
    
    # 2. 获取少量因子数据
    if len(week_end_list) > 0:
        test_date = str(week_end_list[0])[:10]
        print(f"测试日期: {test_date}")
        
        factor_df = DataAPI.MktStockFactorsOneDayGet(
            tradeDate=test_date,
            field=['secID','ticker','tradeDate','MassIndex','RSI']
        )
        print(f"因子数据: {len(factor_df)} 条")
        
        if len(factor_df) > 0:
            print("因子数据样本:")
            print(factor_df.head())
            
            # 3. 获取收益率数据
            returns_df = DataAPI.MktEquwAdjGet(
                beginDate=start_date,
                endDate=end_date,
                field="secID,endDate,chgPct"
            )
            returns_df.rename(columns={'endDate':'tradeDate', 'chgPct':'curr_ret'}, inplace=True)
            returns_df['ticker'] = returns_df['secID'].str.slice(0,6)
            returns_df.sort_values(['ticker', 'tradeDate'], inplace=True)
            returns_df['nxt_ret'] = returns_df.groupby('ticker')['curr_ret'].shift(-1)
            returns_df = returns_df.dropna(subset=['nxt_ret'])
            
            print(f"收益率数据: {len(returns_df)} 条")
            print("收益率数据样本:")
            print(returns_df.head())
            
            # 4. 创建简单的网络动量数据
            # 模拟网络动量因子
            np.random.seed(42)
            network_momentum_df = factor_df[['secID', 'tradeDate']].copy()
            network_momentum_df['period_mean_distance'] = np.random.normal(0, 1, len(network_momentum_df))
            
            print("模拟网络动量数据:")
            print(network_momentum_df.head())
            
            # 5. 测试合并
            print("\n测试数据合并...")
            
            # 确保日期格式一致
            network_momentum_df['tradeDate'] = pd.to_datetime(network_momentum_df['tradeDate'])
            returns_df['tradeDate'] = pd.to_datetime(returns_df['tradeDate'])
            
            print(f"网络动量数据日期范围: {network_momentum_df['tradeDate'].min()} 到 {network_momentum_df['tradeDate'].max()}")
            print(f"收益率数据日期范围: {returns_df['tradeDate'].min()} 到 {returns_df['tradeDate'].max()}")
            
            # 检查共同股票和日期
            common_stocks = set(network_momentum_df['secID']) & set(returns_df['secID'])
            common_dates = set(network_momentum_df['tradeDate']) & set(returns_df['tradeDate'])
            
            print(f"共同股票数量: {len(common_stocks)}")
            print(f"共同日期数量: {len(common_dates)}")
            
            if len(common_stocks) > 0 and len(common_dates) > 0:
                # 合并数据
                merged_df = network_momentum_df.merge(returns_df, on=['secID', 'tradeDate'])
                print(f"合并后数据量: {len(merged_df)}")
                
                if len(merged_df) > 0:
                    print("合并数据样本:")
                    print(merged_df[['secID', 'tradeDate', 'period_mean_distance', 'nxt_ret']].head())
                    
                    # 6. 计算IC
                    if len(merged_df) > 1:
                        ic = merged_df[['period_mean_distance', 'nxt_ret']].corr(method="spearman").iloc[0, 1]
                        print(f"\n测试IC值: {ic:.4f}")
                        
                        print("\n✓ 快速测试成功！所有步骤都能正常工作。")
                        return True
                    else:
                        print("数据量太少，无法计算IC")
                else:
                    print("合并后数据为空")
            else:
                print("没有共同的股票或日期")
    
    print("✗ 快速测试失败")
    return False

if __name__ == "__main__":
    quick_test()
