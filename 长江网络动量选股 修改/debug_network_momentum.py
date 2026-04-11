# -*- coding: utf-8 -*-
"""
调试版本的网络动量策略
"""

import pandas as pd
import numpy as np
from datayes_mysql_api import DataAPI
import traceback

def debug_network_momentum():
    """调试网络动量策略"""
    
    try:
        print("开始调试网络动量策略...")
        
        # 使用较小的日期范围进行测试
        start_date = "2019-01-01"
        end_date = "2019-01-31"
        
        print(f"日期范围: {start_date} 到 {end_date}")
        
        # 1. 获取交易日历
        print("\n1. 获取交易日历...")
        calendar_df = DataAPI.TradeCalGet(exchangeCD="XSHG", beginDate=start_date, endDate=end_date)
        print(f"获取到 {len(calendar_df)} 条交易日历数据")
        
        if len(calendar_df) == 0:
            print("错误: 没有获取到交易日历数据")
            return
        
        week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
        print(f"周末交易日数量: {len(week_end_list)}")
        print(f"周末日期: {week_end_list}")
        
        if len(week_end_list) == 0:
            print("警告: 没有周末交易日，使用所有交易日")
            week_end_list = calendar_df[calendar_df['isOpen']==1]['calendarDate'].values[:5]  # 只取前5个
            print(f"使用的日期: {week_end_list}")
        
        # 2. 获取因子数据
        print("\n2. 获取因子数据...")
        factor_df_list = []
        factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
        
        for i, wnenddate in enumerate(week_end_list):
            print(f"  获取第 {i+1}/{len(week_end_list)} 个日期的数据: {wnenddate}")
            try:
                factor_dfi = DataAPI.MktStockFactorsOneDayGet(
                    tradeDate=str(wnenddate)[:10],  # 确保日期格式正确
                    field=['secID','ticker','tradeDate']+factor_list
                )
                print(f"    获取到 {len(factor_dfi)} 条记录")
                if len(factor_dfi) > 0:
                    factor_df_list.append(factor_dfi)
                else:
                    print(f"    警告: 日期 {wnenddate} 没有数据")
            except Exception as e:
                print(f"    错误: 获取日期 {wnenddate} 的数据失败: {e}")
                continue
        
        if not factor_df_list:
            print("错误: 没有获取到任何因子数据")
            return
        
        print(f"\n成功获取 {len(factor_df_list)} 个日期的因子数据")
        factor_df = pd.concat(factor_df_list, axis=0)
        print(f"合并后总记录数: {len(factor_df)}")
        print("因子数据样本:")
        print(factor_df.head())
        
        # 3. 检查数据质量
        print("\n3. 检查数据质量...")
        print(f"股票数量: {factor_df['secID'].nunique()}")
        print(f"日期数量: {factor_df['tradeDate'].nunique()}")
        print("因子缺失值统计:")
        for col in factor_list:
            missing_count = factor_df[col].isna().sum()
            print(f"  {col}: {missing_count} 个缺失值")
        
        # 4. 简单的因子标准化测试
        print("\n4. 测试因子标准化...")
        factor_df_clean = factor_df.dropna()
        print(f"去除缺失值后记录数: {len(factor_df_clean)}")
        
        if len(factor_df_clean) > 0:
            print("标准化前样本:")
            print(factor_df_clean[factor_list].head())
            
            # 简单标准化
            for col in factor_list:
                factor_df_clean[col] = (factor_df_clean[col] - factor_df_clean[col].mean()) / factor_df_clean[col].std()
            
            print("标准化后样本:")
            print(factor_df_clean[factor_list].head())
        
        print("\n调试完成!")
        
    except Exception as e:
        print(f"调试过程中出错: {e}")
        print("错误详情:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_network_momentum()
