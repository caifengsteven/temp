# -*- coding: utf-8 -*-
"""
优化版本的网络动量选股策略
使用MySQL数据库，优化了性能和输出
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
            print(f"    警告: 没有可用的因子列")
            return pd.Series(dtype=float)

        # 设置secID为索引，只保留因子列
        df1 = df1.set_index('secID')[available_cols].sort_index()

        # 去除包含NaN的行
        df1 = df1.dropna()

        if len(df1) < 2:
            print(f"    警告: 有效数据不足2条 (实际: {len(df1)})")
            return pd.Series(dtype=float)

        print(f"    计算 {len(df1)} 只股票的距离矩阵，使用因子: {available_cols}")

        # 计算欧几里得距离
        df2 = df1.values.repeat([len(df1)] * len(df1), axis=0)
        df3 = np.concatenate([df1] * len(df1))
        distances = np.sqrt(np.sum((df2 - df3) ** 2, axis=1)).reshape(len(df1), -1)

        # 创建距离矩阵
        df4 = pd.DataFrame(distances, index=df1.index, columns=df1.index)

        # 计算每只股票与其他股票的平均距离
        df5 = df4.mean(axis=1)

        print(f"    距离计算完成，平均距离范围: {df5.min():.4f} 到 {df5.max():.4f}")

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

def calculate_portfolio_returns(factor_return_df):
    """
    基于网络动量因子构建投资组合并计算收益率

    Args:
        factor_return_df: 包含因子和收益率的DataFrame

    Returns:
        portfolio_returns: 投资组合收益率时间序列
    """
    try:
        print("  构建多空投资组合...")

        portfolio_returns = []

        # 按日期分组进行投资组合构建
        for trade_date, group in factor_return_df.groupby('tradeDate'):
            group = group.dropna(subset=['period_mean_distance', 'nxt_ret'])

            if len(group) < 20:  # 至少需要20只股票
                continue

            # 按网络动量因子排序
            group = group.sort_values('period_mean_distance')

            # 选择前20%和后20%的股票
            n_stocks = len(group)
            n_select = max(10, n_stocks // 5)  # 至少选择10只股票

            # 多头：网络距离最小的股票（更相似，可能表现更好）
            long_stocks = group.head(n_select)
            # 空头：网络距离最大的股票（更孤立，可能表现更差）
            short_stocks = group.tail(n_select)

            # 计算多空组合收益率
            long_return = long_stocks['nxt_ret'].mean()
            short_return = short_stocks['nxt_ret'].mean()
            portfolio_return = long_return - short_return  # 多空收益

            portfolio_returns.append({
                'tradeDate': trade_date,
                'portfolio_return': portfolio_return,
                'long_return': long_return,
                'short_return': short_return,
                'n_stocks': n_stocks,
                'n_long': len(long_stocks),
                'n_short': len(short_stocks)
            })

        portfolio_df = pd.DataFrame(portfolio_returns)

        if len(portfolio_df) > 0:
            # 计算累计收益率
            portfolio_df = portfolio_df.sort_values('tradeDate')
            portfolio_df['cum_return'] = (1 + portfolio_df['portfolio_return']).cumprod()
            portfolio_df['cum_long'] = (1 + portfolio_df['long_return']).cumprod()
            portfolio_df['cum_short'] = (1 + portfolio_df['short_return']).cumprod()

            print(f"    平均多头收益: {portfolio_df['long_return'].mean():.4f}")
            print(f"    平均空头收益: {portfolio_df['short_return'].mean():.4f}")
            print(f"    平均多空收益: {portfolio_df['portfolio_return'].mean():.4f}")
            print(f"    多空收益胜率: {(portfolio_df['portfolio_return'] > 0).mean():.2%}")

        return portfolio_df

    except Exception as e:
        print(f"  投资组合构建失败: {e}")
        return None

def plot_equity_curve(portfolio_returns, period_ic):
    """
    绘制权益曲线和IC时间序列

    Args:
        portfolio_returns: 投资组合收益率DataFrame
        period_ic: IC时间序列
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 1. 权益曲线
        ax1.plot(portfolio_returns['tradeDate'], portfolio_returns['cum_return'],
                label='多空组合', linewidth=2, color='red')
        ax1.plot(portfolio_returns['tradeDate'], portfolio_returns['cum_long'],
                label='多头组合', linewidth=1, color='green', alpha=0.7)
        ax1.plot(portfolio_returns['tradeDate'], portfolio_returns['cum_short'],
                label='空头组合', linewidth=1, color='blue', alpha=0.7)

        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('网络动量策略权益曲线', fontsize=14, fontweight='bold')
        ax1.set_ylabel('累计收益率', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        # 2. IC时间序列
        ic_dates = pd.to_datetime(period_ic.index)
        ax2.plot(ic_dates, period_ic.values, marker='o', linewidth=1, markersize=4, color='purple')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=period_ic.mean(), color='red', linestyle='--', alpha=0.7,
                   label=f'平均IC: {period_ic.mean():.4f}')

        ax2.set_title('IC时间序列', fontsize=14, fontweight='bold')
        ax2.set_ylabel('IC值', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        plt.tight_layout()
        plt.xticks(rotation=45)

        # 保存图片
        plt.savefig('网络动量策略权益曲线.png', dpi=300, bbox_inches='tight')
        print("  ✓ 权益曲线已保存为: 网络动量策略权益曲线.png")

        # 显示图片
        plt.show()

    except Exception as e:
        print(f"  权益曲线绘制失败: {e}")

def calculate_strategy_metrics(portfolio_returns):
    """
    计算策略表现指标

    Args:
        portfolio_returns: 投资组合收益率DataFrame

    Returns:
        metrics: 策略指标字典
    """
    try:
        returns = portfolio_returns['portfolio_return']
        cum_returns = portfolio_returns['cum_return']

        # 基本统计
        total_return = cum_returns.iloc[-1] - 1
        annualized_return = (cum_returns.iloc[-1] ** (252 / len(returns))) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # 最大回撤
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (returns > 0).mean()

        # 收益风险比
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        metrics = {
            '总收益率': f"{total_return:.2%}",
            '年化收益率': f"{annualized_return:.2%}",
            '年化波动率': f"{volatility:.2%}",
            '夏普比率': f"{sharpe_ratio:.4f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '胜率': f"{win_rate:.2%}",
            '盈亏比': f"{profit_loss_ratio:.4f}",
            '交易次数': f"{len(returns)}次"
        }

        return metrics

    except Exception as e:
        print(f"  策略指标计算失败: {e}")
        return {}

def run_network_momentum_strategy():
    """运行网络动量策略"""
    
    print("=" * 60)
    print("网络动量选股策略 - 基于MySQL数据库")
    print("=" * 60)
    
    # 使用更长的时间范围进行完整分析
    start_date = "2015-01-01"
    end_date = "2019-08-23"
    
    print(f"分析期间: {start_date} 到 {end_date}")
    
    # 1. 获取交易日历
    print("\n第一步: 获取交易日历...")
    start_time = time.time()
    
    calendar_df = DataAPI.TradeCalGet(exchangeCD="XSHG", beginDate=start_date, endDate=end_date)
    week_end_list = calendar_df[calendar_df['isWeekEnd']==1]['calendarDate'].values
    month_end_list = calendar_df[calendar_df['isMonthEnd']==1]['calendarDate'].values
    trade_date_list = calendar_df[calendar_df['isOpen']==1]['calendarDate'].values
    
    print(f"✓ 获取到 {len(week_end_list)} 个周末交易日")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 2. 获取因子数据
    print("\n第二步: 获取股票因子数据...")
    start_time = time.time()
    
    factor_df_list = []
    factor_list = ['MassIndex','KDJ_J','RSI','CCI10','CMO','MFI']
    
    # 处理更多日期以获得更完整的分析
    max_dates = min(100, len(week_end_list))  # 最多处理100个周末（约2年数据）
    week_end_list_limited = week_end_list[:max_dates]
    
    print(f"  处理 {len(week_end_list_limited)} 个周末日期...")
    
    for i, wnenddate in enumerate(week_end_list_limited):
        if i % 10 == 0:  # 每10个日期显示一次进度
            print(f"  进度: {i+1}/{len(week_end_list_limited)} ({(i+1)/len(week_end_list_limited)*100:.1f}%)")
        
        try:
            factor_dfi = DataAPI.MktStockFactorsOneDayGet(
                tradeDate=str(wnenddate)[:10],
                field=['secID','ticker','tradeDate']+factor_list
            )
            if len(factor_dfi) > 0:
                factor_df_list.append(factor_dfi)
        except Exception as e:
            print(f"    警告: 日期 {wnenddate} 数据获取失败: {e}")
            continue
    
    if not factor_df_list:
        print("错误: 没有获取到因子数据")
        return None
    
    factor_df = pd.concat(factor_df_list, axis=0)
    print(f"✓ 获取到 {len(factor_df)} 条因子数据")
    print(f"  股票数量: {factor_df['secID'].nunique()}")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 3. 因子标准化
    print("\n第三步: 因子标准化...")
    start_time = time.time()

    # 检查数据质量
    print(f"  标准化前数据量: {len(factor_df)}")
    print(f"  缺失值统计:")
    for col in factor_list:
        missing_count = factor_df[col].isna().sum()
        print(f"    {col}: {missing_count} 个缺失值 ({missing_count/len(factor_df)*100:.1f}%)")

    # 先去除缺失值过多的记录
    factor_df_clean = factor_df.dropna(subset=factor_list)
    print(f"  去除缺失值后数据量: {len(factor_df_clean)}")

    if len(factor_df_clean) == 0:
        print("  错误: 去除缺失值后没有数据，尝试使用填充方法...")
        # 使用前向填充处理缺失值
        factor_df_clean = factor_df.copy()
        for col in factor_list:
            factor_df_clean[col] = factor_df_clean.groupby('secID')[col].fillna(method='ffill')
        factor_df_clean = factor_df_clean.dropna(subset=factor_list)
        print(f"  填充后数据量: {len(factor_df_clean)}")

    if len(factor_df_clean) == 0:
        print("  错误: 仍然没有有效数据，跳过标准化...")
        factor_df1 = factor_df.copy()
    else:
        # 按股票分组进行标准化，但要求每只股票至少有5个观测值
        def safe_cal_std_factor(group):
            if len(group) < 5:  # 如果观测值太少，跳过标准化
                return group
            return cal_std_factor(group, min(52, len(group)))

        factor_df1 = factor_df_clean.groupby(['secID', 'ticker'], group_keys=False).apply(safe_cal_std_factor)
        factor_df1 = factor_df1.dropna(subset=factor_list)

    factor_df1 = factor_df1[['secID', 'ticker', 'tradeDate', 'KDJ_J', 'RSI', 'CCI10', 'MFI', 'MassIndex', 'CMO']]

    print(f"✓ 标准化完成，剩余 {len(factor_df1)} 条有效数据")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")

    if len(factor_df1) == 0:
        print("错误: 标准化后没有有效数据，无法继续计算")
        return None
    
    # 4. 计算网络动量
    print("\n第四步: 计算网络动量...")
    start_time = time.time()
    
    print("  计算股票间距离矩阵...")

    # 重新设计距离计算，保留tradeDate信息
    distance_results = []
    unique_dates = factor_df1['tradeDate'].unique()

    print(f"  需要处理 {len(unique_dates)} 个日期")

    for i, trade_date in enumerate(unique_dates):
        if i % 10 == 0:  # 每10个日期显示一次进度
            print(f"    距离计算进度: {i+1}/{len(unique_dates)} ({(i+1)/len(unique_dates)*100:.1f}%)")

        date_data = factor_df1[factor_df1['tradeDate'] == trade_date]
        print(f"    日期 {trade_date}: {len(date_data)} 只股票")

        if len(date_data) > 1:  # 至少需要2只股票才能计算距离
            distances = cal_distance(date_data)
            if len(distances) > 0:
                distance_df = pd.DataFrame({
                    'secID': distances.index,
                    'tradeDate': trade_date,
                    'mean_distance': distances.values
                })
                distance_results.append(distance_df)
                print(f"    ✓ 成功计算 {len(distances)} 只股票的距离")
            else:
                print(f"    ✗ 距离计算失败")
        else:
            print(f"    ✗ 股票数量不足 ({len(date_data)} < 2)")

    if not distance_results:
        print("  错误: 无法计算距离矩阵")
        return None

    stock_distance_matrix = pd.concat(distance_results, ignore_index=True)
    print(f"  距离矩阵计算完成，包含列: {list(stock_distance_matrix.columns)}")
    print(f"  总记录数: {len(stock_distance_matrix)}")

    print("  计算期间平均距离...")
    stock_distance_matrix1 = stock_distance_matrix.groupby('secID', group_keys=False).apply(
        lambda x: cal_period_mean_distance(x, 4)
    ).dropna().reset_index(drop=True)
    
    print(f"✓ 网络动量计算完成，得到 {len(stock_distance_matrix1)} 条记录")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 5. 获取收益率数据进行回测
    print("\n第五步: 获取收益率数据...")
    start_time = time.time()
    
    # 分段获取收益率数据（更长时间范围）
    print("  分段获取收益率数据...")
    bt_mret_df1 = DataAPI.MktEquwAdjGet(beginDate=start_date, endDate="2015-12-31", field="secID,endDate,chgPct")
    print(f"    2015年数据: {len(bt_mret_df1)} 条")

    bt_mret_df2 = DataAPI.MktEquwAdjGet(beginDate='2016-01-01', endDate="2016-12-31", field="secID,endDate,chgPct")
    print(f"    2016年数据: {len(bt_mret_df2)} 条")

    bt_mret_df3 = DataAPI.MktEquwAdjGet(beginDate='2017-01-01', endDate="2017-12-31", field="secID,endDate,chgPct")
    print(f"    2017年数据: {len(bt_mret_df3)} 条")

    bt_mret_df4 = DataAPI.MktEquwAdjGet(beginDate='2018-01-01', endDate="2018-12-31", field="secID,endDate,chgPct")
    print(f"    2018年数据: {len(bt_mret_df4)} 条")

    bt_mret_df5 = DataAPI.MktEquwAdjGet(beginDate='2019-01-01', endDate=end_date, field="secID,endDate,chgPct")
    print(f"    2019年数据: {len(bt_mret_df5)} 条")

    bt_mret_df = pd.concat([bt_mret_df1, bt_mret_df2, bt_mret_df3, bt_mret_df4, bt_mret_df5], ignore_index=True)
    
    bt_mret_df.rename(columns={'endDate':'tradeDate', 'chgPct':'curr_ret'}, inplace=True)
    bt_mret_df['ticker'] = bt_mret_df['secID'].str.slice(0,6)
    bt_mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    bt_mret_df['nxt_ret'] = bt_mret_df.groupby('ticker')['curr_ret'].shift(-1)
    bt_mret_df = bt_mret_df.dropna(subset=['nxt_ret'])
    
    print(f"✓ 获取到 {len(bt_mret_df)} 条收益率数据")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 6. 因子测试
    print("\n第六步: 因子有效性测试...")
    start_time = time.time()

    print(f"  网络动量数据列: {list(stock_distance_matrix1.columns)}")
    print(f"  收益率数据列: {list(bt_mret_df.columns)}")

    back_test_date = '2015-06-01'  # 调整为更早的日期，确保有足够的回测数据

    # 检查tradeDate列是否存在
    if 'tradeDate' not in stock_distance_matrix1.columns:
        print("  错误: 网络动量数据中缺少tradeDate列")
        print(f"  可用列: {list(stock_distance_matrix1.columns)}")
        return None

    # 确保日期格式一致
    stock_distance_matrix1['tradeDate'] = pd.to_datetime(stock_distance_matrix1['tradeDate'])
    bt_mret_df['tradeDate'] = pd.to_datetime(bt_mret_df['tradeDate'])
    back_test_date = pd.to_datetime(back_test_date)

    print(f"  过滤前网络动量数据: {len(stock_distance_matrix1)} 条")
    print(f"  回测开始日期: {back_test_date}")
    print(f"  网络动量数据日期范围: {stock_distance_matrix1['tradeDate'].min()} 到 {stock_distance_matrix1['tradeDate'].max()}")

    stock_distance_matrix2 = stock_distance_matrix1[stock_distance_matrix1['tradeDate'] >= back_test_date]
    print(f"  过滤后网络动量数据: {len(stock_distance_matrix2)} 条")
    
    # 调试合并问题
    print(f"  网络动量数据样本:")
    print(stock_distance_matrix2[['secID', 'tradeDate']].head())
    print(f"  收益率数据样本:")
    print(bt_mret_df[['secID', 'tradeDate']].head())

    print(f"  网络动量数据日期范围: {stock_distance_matrix2['tradeDate'].min()} 到 {stock_distance_matrix2['tradeDate'].max()}")
    print(f"  收益率数据日期范围: {bt_mret_df['tradeDate'].min()} 到 {bt_mret_df['tradeDate'].max()}")

    # 检查共同的股票和日期
    common_stocks = set(stock_distance_matrix2['secID']) & set(bt_mret_df['secID'])
    common_dates = set(stock_distance_matrix2['tradeDate']) & set(bt_mret_df['tradeDate'])

    print(f"  共同股票数量: {len(common_stocks)}")
    print(f"  共同日期数量: {len(common_dates)}")

    if len(common_stocks) == 0:
        print("  错误: 没有共同的股票代码")
        print(f"  网络动量股票样本: {list(stock_distance_matrix2['secID'].head())}")
        print(f"  收益率股票样本: {list(bt_mret_df['secID'].head())}")
        return None

    if len(common_dates) == 0:
        print("  错误: 没有共同的日期")
        return None

    factor_rtn_df = stock_distance_matrix2.merge(bt_mret_df, on=['secID', 'tradeDate'])

    if len(factor_rtn_df) == 0:
        print("  警告: 因子和收益率数据合并后为空")
        return None
    
    print(f"✓ 合并后数据量: {len(factor_rtn_df)} 条")
    
    # 计算IC
    period_ic = factor_rtn_df.groupby('tradeDate').apply(
        lambda x: x[['period_mean_distance', 'nxt_ret']].corr(method="spearman").iloc[0, 1] if len(x) > 1 else np.nan
    ).dropna()
    
    if len(period_ic) == 0:
        print("警告: 无法计算IC值")
        return None
    
    ic = period_ic.mean()
    std = period_ic.std()
    icir = ic / std if std != 0 else 0
    ic_t = stats.ttest_1samp(period_ic, 0)[0] if len(period_ic) > 1 else 0
    
    ic_summary = pd.DataFrame([ic, std, icir, ic_t], 
                             index=['IC均值', 'IC波动率', 'ICIR', 't值'], 
                             columns=['网络动量因子']).T.round(4)
    
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    # 7. 结果展示
    print("\n" + "=" * 60)
    print("策略结果汇总")
    print("=" * 60)
    
    print(f"分析期间: {start_date} 到 {end_date}")
    print(f"处理的周末数量: {len(week_end_list_limited)}")
    print(f"股票数量: {factor_df['secID'].nunique()}")
    print(f"有效IC期数: {len(period_ic)}")
    
    print("\nIC统计结果:")
    print(ic_summary)
    
    print(f"\nIC时间序列统计:")
    print(f"  最大IC: {period_ic.max():.4f}")
    print(f"  最小IC: {period_ic.min():.4f}")
    print(f"  IC>0的比例: {(period_ic > 0).mean():.2%}")

    # 7. 构建投资组合和计算收益率
    print("\n第七步: 构建投资组合...")
    start_time = time.time()

    portfolio_returns = calculate_portfolio_returns(factor_rtn_df)

    if portfolio_returns is not None and len(portfolio_returns) > 0:
        print(f"✓ 投资组合构建完成，得到 {len(portfolio_returns)} 期收益")
        print(f"  处理时间: {time.time() - start_time:.2f}秒")

        # 8. 绘制权益曲线
        print("\n第八步: 生成权益曲线...")
        plot_equity_curve(portfolio_returns, period_ic)

        # 计算策略表现指标
        strategy_metrics = calculate_strategy_metrics(portfolio_returns)
        print("\n策略表现指标:")
        for metric, value in strategy_metrics.items():
            print(f"  {metric}: {value}")
    else:
        print("  警告: 无法构建投资组合")

    print("\n策略运行完成！")
    
    return {
        'factor_data': stock_distance_matrix1,
        'ic_summary': ic_summary,
        'factor_return_data': factor_rtn_df,
        'period_ic': period_ic
    }

if __name__ == "__main__":
    try:
        results = run_network_momentum_strategy()
        if results:
            print("\n✓ 策略执行成功！")
        else:
            print("\n✗ 策略执行失败！")
    except Exception as e:
        print(f"\n✗ 策略执行出错: {e}")
        import traceback
        traceback.print_exc()
