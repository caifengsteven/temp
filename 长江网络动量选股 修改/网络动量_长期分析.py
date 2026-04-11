# -*- coding: utf-8 -*-
"""
网络动量选股策略 - 长期分析版本
使用MySQL数据库，分析2015-2019年完整数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

# 导入MySQL版本的DataAPI
from datayes_mysql_api import DataAPI
import lib.quant_util as quant_util

# 设置中文字体
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei']

def cal_std_factor(df, step=52):
    """对指标值进行时间序列标准化"""
    factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
    df = df.copy()
    df = df.sort_values('tradeDate')
    
    for f in factor_list:
        if f in df.columns:
            # 使用更小的min_periods，避免数据全部变成NaN
            min_periods = min(5, len(df) // 2)
            rolling_mean = df[f].rolling(step, min_periods=min_periods).mean()
            rolling_std = df[f].rolling(step, min_periods=min_periods).std()
            
            # 避免除以0
            rolling_std = rolling_std.replace(0, np.nan)
            df[f] = (df[f] - rolling_mean) / rolling_std
    
    return df

def cal_distance(df):
    """计算每个节点与其他节点的平均距离"""
    try:
        df1 = df.copy()
        
        # 只保留数值列进行距离计算
        factor_cols = ['KDJ_J', 'RSI', 'CCI10', 'MFI', 'MassIndex', 'CMO']
        available_cols = [col for col in factor_cols if col in df1.columns]
        
        if len(available_cols) == 0:
            return pd.Series(dtype=float)
        
        # 设置secID为索引，只保留因子列
        df1 = df1.set_index('secID')[available_cols].sort_index()
        
        # 去除包含NaN的行
        df1 = df1.dropna()
        
        if len(df1) < 2:
            return pd.Series(dtype=float)
        
        # 计算欧几里得距离
        df2 = df1.values.repeat([len(df1)] * len(df1), axis=0)
        df3 = np.concatenate([df1] * len(df1))
        distances = np.sqrt(np.sum((df2 - df3) ** 2, axis=1)).reshape(len(df1), -1)
        
        # 创建距离矩阵
        df4 = pd.DataFrame(distances, index=df1.index, columns=df1.index)
        
        # 计算每只股票与其他股票的平均距离
        df5 = df4.mean(axis=1)
        
        del df1, df2, df3, df4
        return df5
        
    except Exception as e:
        print(f"    距离计算出错: {e}")
        return pd.Series(dtype=float)

def cal_period_mean_distance(df, period_step=4):
    """计算过去四周网络距离的平均值"""
    df = df.copy()
    if 'tradeDate' in df.columns:
        df = df.sort_values('tradeDate')
    df['period_mean_distance'] = df['mean_distance'].rolling(period_step).mean()
    return df

def run_long_term_analysis():
    """运行长期网络动量分析"""
    
    print("=" * 80)
    print("网络动量选股策略 - 长期分析 (2015-2019)")
    print("=" * 80)
    
    # 长期分析时间范围
    start_date = "2015-01-01"
    end_date = "2019-08-23"
    
    print(f"分析期间: {start_date} 到 {end_date} (约4.7年)")
    
    # 1. 获取交易日历
    print("\n第一步: 获取交易日历...")
    start_time = time.time()
    
    calendar_df = DataAPI.TradeCalGet(exchangeCD="XSHG", beginDate=start_date, endDate=end_date)
    week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
    
    print(f"✓ 获取到 {len(week_end_list)} 个周末交易日")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 2. 分批处理因子数据
    print("\n第二步: 分批获取股票因子数据...")
    start_time = time.time()
    
    factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
    
    # 按年份分批处理
    years = ['2015', '2016', '2017', '2018', '2019']
    all_factor_data = []
    
    for year in years:
        print(f"\n  处理 {year} 年数据...")
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31" if year != "2019" else end_date
        
        # 获取该年的周末日期
        year_calendar = calendar_df[
            (pd.to_datetime(calendar_df['calendarDate']) >= pd.to_datetime(year_start)) &
            (pd.to_datetime(calendar_df['calendarDate']) <= pd.to_datetime(year_end)) &
            (calendar_df['isWeekEnd'] == 1)
        ]
        year_week_ends = year_calendar['calendarDate'].values
        
        print(f"    {year}年周末数量: {len(year_week_ends)}")
        
        year_factor_data = []
        for i, wnenddate in enumerate(year_week_ends):
            if i % 10 == 0:
                print(f"    进度: {i+1}/{len(year_week_ends)} ({(i+1)/len(year_week_ends)*100:.1f}%)")
            
            try:
                factor_dfi = DataAPI.MktStockFactorsOneDayGet(
                    tradeDate=str(wnenddate)[:10],
                    field=['secID','ticker','tradeDate']+factor_list
                )
                if len(factor_dfi) > 0:
                    year_factor_data.append(factor_dfi)
            except Exception as e:
                print(f"      警告: 日期 {wnenddate} 数据获取失败: {e}")
                continue
        
        if year_factor_data:
            year_df = pd.concat(year_factor_data, axis=0)
            all_factor_data.append(year_df)
            print(f"    ✓ {year}年获取到 {len(year_df)} 条因子数据")
        else:
            print(f"    ✗ {year}年没有获取到数据")
    
    if not all_factor_data:
        print("错误: 没有获取到任何因子数据")
        return None
    
    factor_df = pd.concat(all_factor_data, axis=0)
    print(f"\n✓ 总计获取到 {len(factor_df)} 条因子数据")
    print(f"  股票数量: {factor_df['secID'].nunique()}")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 3. 因子标准化
    print("\n第三步: 因子标准化...")
    start_time = time.time()
    
    # 检查数据质量
    print(f"  标准化前数据量: {len(factor_df)}")
    for col in factor_list:
        missing_count = factor_df[col].isna().sum()
        print(f"    {col}: {missing_count} 个缺失值 ({missing_count/len(factor_df)*100:.1f}%)")
    
    # 处理缺失值
    factor_df_clean = factor_df.dropna(subset=factor_list)
    print(f"  去除缺失值后数据量: {len(factor_df_clean)}")
    
    if len(factor_df_clean) == 0:
        print("  错误: 去除缺失值后没有数据")
        return None
    
    # 按股票分组进行标准化
    def safe_cal_std_factor(group):
        if len(group) < 10:  # 要求每只股票至少有10个观测值
            return group
        return cal_std_factor(group, min(52, len(group)))
    
    factor_df1 = factor_df_clean.groupby(['secID', 'ticker'], group_keys=False).apply(safe_cal_std_factor)
    factor_df1 = factor_df1.dropna(subset=factor_list)
    factor_df1 = factor_df1[['secID', 'ticker', 'tradeDate', 'KDJ_J', 'RSI', 'CCI10', 'MFI', 'MassIndex', 'CMO']]
    
    print(f"✓ 标准化完成，剩余 {len(factor_df1)} 条有效数据")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    if len(factor_df1) == 0:
        print("错误: 标准化后没有有效数据")
        return None
    
    # 4. 计算网络动量
    print("\n第四步: 计算网络动量...")
    start_time = time.time()
    
    print("  计算股票间距离矩阵...")
    distance_results = []
    unique_dates = factor_df1['tradeDate'].unique()
    
    print(f"  需要处理 {len(unique_dates)} 个日期")
    
    for i, trade_date in enumerate(unique_dates):
        if i % 20 == 0:  # 每20个日期显示一次进度
            print(f"    距离计算进度: {i+1}/{len(unique_dates)} ({(i+1)/len(unique_dates)*100:.1f}%)")
        
        date_data = factor_df1[factor_df1['tradeDate'] == trade_date]
        
        if len(date_data) > 10:  # 至少需要10只股票才能计算距离
            distances = cal_distance(date_data)
            if len(distances) > 0:
                distance_df = pd.DataFrame({
                    'secID': distances.index,
                    'tradeDate': trade_date,
                    'mean_distance': distances.values
                })
                distance_results.append(distance_df)
    
    if not distance_results:
        print("  错误: 无法计算距离矩阵")
        return None
    
    stock_distance_matrix = pd.concat(distance_results, ignore_index=True)
    print(f"  距离矩阵计算完成，总记录数: {len(stock_distance_matrix)}")
    
    print("  计算期间平均距离...")
    stock_distance_matrix1 = stock_distance_matrix.groupby('secID', group_keys=False).apply(
        lambda x: cal_period_mean_distance(x, 4)
    ).dropna().reset_index(drop=True)
    
    print(f"✓ 网络动量计算完成，得到 {len(stock_distance_matrix1)} 条记录")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    return {
        'factor_data': factor_df1,
        'network_momentum': stock_distance_matrix1,
        'analysis_period': f"{start_date} 到 {end_date}",
        'total_stocks': factor_df['secID'].nunique(),
        'total_periods': len(unique_dates)
    }

if __name__ == "__main__":
    try:
        print("开始长期网络动量分析...")
        results = run_long_term_analysis()
        
        if results:
            print("\n" + "=" * 80)
            print("长期分析完成！")
            print("=" * 80)
            print(f"分析期间: {results['analysis_period']}")
            print(f"总股票数: {results['total_stocks']}")
            print(f"总期数: {results['total_periods']}")
            print(f"网络动量记录: {len(results['network_momentum'])}")
            print("\n✓ 长期分析执行成功！")
            
            # 保存结果
            results['network_momentum'].to_csv('网络动量_长期分析结果.csv', index=False, encoding='utf-8-sig')
            print("✓ 结果已保存到: 网络动量_长期分析结果.csv")
            
        else:
            print("\n✗ 长期分析执行失败！")
            
    except Exception as e:
        print(f"\n✗ 长期分析执行出错: {e}")
        import traceback
        traceback.print_exc()
