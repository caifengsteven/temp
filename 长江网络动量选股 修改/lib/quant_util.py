# -*- coding: utf-8 -*-
"""
量化工具模块
用于网络动量策略的回测和分析功能
"""

import pandas as pd
import numpy as np
from scipy import stats

def simple_group_backtest(factor_df, return_df, factor_col, return_col, ngrp=5):
    """
    简单的分组回测
    
    Args:
        factor_df: 因子数据
        return_df: 收益率数据
        factor_col: 因子列名
        return_col: 收益率列名
        ngrp: 分组数量
    
    Returns:
        回测结果DataFrame
    """
    try:
        # 合并因子和收益率数据
        merged_df = factor_df.merge(return_df, on=['secID', 'tradeDate'], how='inner')
        
        if len(merged_df) == 0:
            print("警告：因子和收益率数据合并后为空")
            return pd.DataFrame()
        
        # 按日期分组，计算因子分位数
        def assign_groups(group):
            group = group.dropna(subset=[factor_col])
            if len(group) == 0:
                return group
            
            # 计算分位数
            group['factor_rank'] = group[factor_col].rank(method='first')
            group['group'] = pd.cut(group['factor_rank'], bins=ngrp, labels=False)
            return group
        
        grouped_df = merged_df.groupby('tradeDate').apply(assign_groups).reset_index(drop=True)
        
        # 计算每组的平均收益率
        group_returns = grouped_df.groupby(['tradeDate', 'group'])[return_col].mean().reset_index()
        
        # 计算累计收益率
        result_list = []
        for group_id in range(ngrp):
            group_data = group_returns[group_returns['group'] == group_id].copy()
            if len(group_data) > 0:
                group_data = group_data.sort_values('tradeDate')
                group_data['cum_ret'] = (1 + group_data[return_col]).cumprod()
                group_data['group'] = group_id
                result_list.append(group_data)
        
        if result_list:
            result_df = pd.concat(result_list, ignore_index=True)
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"分组回测出错: {e}")
        return pd.DataFrame()

def long_short_backtest(factor_df, return_df, factor_col, return_col, direction=1, commission=0.0):
    """
    多空回测
    
    Args:
        factor_df: 因子数据
        return_df: 收益率数据
        factor_col: 因子列名
        return_col: 收益率列名
        direction: 方向 (1为正向，-1为反向)
        commission: 手续费
    
    Returns:
        (回测结果DataFrame, 详细交易DataFrame)
    """
    try:
        # 合并因子和收益率数据
        merged_df = factor_df.merge(return_df, on=['secID', 'tradeDate'], how='inner')
        
        if len(merged_df) == 0:
            print("警告：因子和收益率数据合并后为空")
            return pd.DataFrame(), pd.DataFrame()
        
        # 按日期分组，选择多空股票
        def select_long_short(group):
            group = group.dropna(subset=[factor_col])
            if len(group) < 10:  # 至少需要10只股票
                return pd.DataFrame()
            
            # 排序
            group = group.sort_values(factor_col, ascending=(direction == 1))
            
            # 选择前20%和后20%
            n_stocks = len(group)
            n_select = max(1, n_stocks // 5)
            
            long_stocks = group.head(n_select).copy()
            short_stocks = group.tail(n_select).copy()
            
            long_stocks['position'] = 1
            short_stocks['position'] = -1
            
            return pd.concat([long_stocks, short_stocks])
        
        portfolio_df = merged_df.groupby('tradeDate').apply(select_long_short).reset_index(drop=True)
        
        if len(portfolio_df) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # 计算每日组合收益率
        daily_returns = portfolio_df.groupby('tradeDate').apply(
            lambda x: (x[return_col] * x['position']).mean()
        ).reset_index()
        daily_returns.columns = ['tradeDate', 'period_ret']
        
        # 扣除手续费
        daily_returns['period_ret'] = daily_returns['period_ret'] - commission
        
        # 计算累计收益率
        daily_returns = daily_returns.sort_values('tradeDate')
        daily_returns['cum_ret'] = (1 + daily_returns['period_ret']).cumprod()
        
        return daily_returns, portfolio_df
        
    except Exception as e:
        print(f"多空回测出错: {e}")
        return pd.DataFrame(), pd.DataFrame()

def signal_grouping(factor_df, factor_col, ngrp=5):
    """
    信号分组
    
    Args:
        factor_df: 因子数据
        factor_col: 因子列名
        ngrp: 分组数量
    
    Returns:
        带有分组信息的DataFrame
    """
    try:
        result_df = factor_df.copy()
        
        def assign_groups(group):
            group = group.dropna(subset=[factor_col])
            if len(group) == 0:
                return group
            
            # 计算分位数
            group['factor_rank'] = group[factor_col].rank(method='first')
            group['group'] = pd.cut(group['factor_rank'], bins=ngrp, labels=False)
            return group
        
        result_df = result_df.groupby('tradeDate').apply(assign_groups).reset_index(drop=True)
        return result_df
        
    except Exception as e:
        print(f"信号分组出错: {e}")
        return factor_df

def netralize_dframe(factor_df, factor_cols, exclude_style=None):
    """
    因子中性化（简化版本）
    
    Args:
        factor_df: 因子数据
        factor_cols: 需要中性化的因子列
        exclude_style: 排除的风格因子
    
    Returns:
        中性化后的DataFrame
    """
    try:
        # 这里是一个简化的实现
        # 在实际应用中，您需要实现完整的中性化逻辑
        result_df = factor_df.copy()
        
        for col in factor_cols:
            if col in result_df.columns:
                # 简单的标准化处理
                result_df[col] = (result_df[col] - result_df[col].mean()) / result_df[col].std()
        
        return result_df
        
    except Exception as e:
        print(f"因子中性化出错: {e}")
        return factor_df

def calculate_ic(factor_df, return_df, factor_col, return_col):
    """
    计算IC值
    
    Args:
        factor_df: 因子数据
        return_df: 收益率数据
        factor_col: 因子列名
        return_col: 收益率列名
    
    Returns:
        IC序列
    """
    try:
        merged_df = factor_df.merge(return_df, on=['secID', 'tradeDate'], how='inner')
        
        if len(merged_df) == 0:
            return pd.Series()
        
        # 按日期计算IC
        ic_series = merged_df.groupby('tradeDate').apply(
            lambda x: x[[factor_col, return_col]].corr(method='spearman').iloc[0, 1]
        )
        
        return ic_series
        
    except Exception as e:
        print(f"计算IC出错: {e}")
        return pd.Series()

# 兼容性函数
def cal_std_factor(df, step=52):
    """对指标值进行时间序列标准化"""
    factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
    df = df.copy()
    df = df.sort_values('tradeDate')
    for f in factor_list:
        if f in df.columns:
            df[f] = (df[f] - df[f].rolling(step,min_periods=30).mean()) / df[f].rolling(step,min_periods=30).std()
    return df
