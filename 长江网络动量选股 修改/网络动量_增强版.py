# -*- coding: utf-8 -*-
"""
网络动量选股策略 - 增强版
1. 扩展到2023年 (9年数据)
2. 提高年化收益率
3. 多种策略优化
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

# 设置中文字体
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei']

def calculate_enhanced_portfolio_returns(factor_return_df, strategy_type='enhanced_long_short'):
    """
    增强版投资组合构建
    
    Args:
        factor_return_df: 包含因子和收益率的DataFrame
        strategy_type: 策略类型
            - 'enhanced_long_short': 增强多空策略 (top/bottom 10%, 2x leverage)
            - 'momentum_only': 纯多头动量策略
            - 'market_neutral': 市场中性策略
    """
    try:
        print(f"  构建{strategy_type}投资组合...")
        
        portfolio_returns = []
        
        for trade_date, group in factor_return_df.groupby('tradeDate'):
            group = group.dropna(subset=['period_mean_distance', 'nxt_ret'])
            
            if len(group) < 100:  # 至少需要100只股票
                continue
            
            # 按网络动量因子排序
            group = group.sort_values('period_mean_distance')
            n_stocks = len(group)
            
            if strategy_type == 'enhanced_long_short':
                # 增强多空策略：选择前10%和后10%，使用2倍杠杆
                n_select = max(20, n_stocks // 10)  # 前后10%
                
                long_stocks = group.head(n_select)
                short_stocks = group.tail(n_select)
                
                long_return = long_stocks['nxt_ret'].mean()
                short_return = short_stocks['nxt_ret'].mean()
                
                # 2倍杠杆
                leverage = 2.0
                portfolio_return = leverage * (long_return - short_return)
                
                portfolio_returns.append({
                    'tradeDate': trade_date,
                    'portfolio_return': portfolio_return,
                    'long_return': long_return,
                    'short_return': short_return,
                    'leverage': leverage,
                    'n_stocks': n_stocks,
                    'n_long': len(long_stocks),
                    'n_short': len(short_stocks)
                })
                
            elif strategy_type == 'momentum_only':
                # 纯多头动量策略：只做多网络距离最小的股票
                n_select = max(50, n_stocks // 5)  # 前20%
                
                long_stocks = group.head(n_select)
                portfolio_return = long_stocks['nxt_ret'].mean()
                
                portfolio_returns.append({
                    'tradeDate': trade_date,
                    'portfolio_return': portfolio_return,
                    'n_stocks': n_stocks,
                    'n_selected': len(long_stocks)
                })
                
            elif strategy_type == 'market_neutral':
                # 市场中性策略：动态调整权重
                n_select = max(30, n_stocks // 8)  # 前后12.5%
                
                long_stocks = group.head(n_select)
                short_stocks = group.tail(n_select)
                
                # 根据因子强度调整权重
                long_weights = 1 / long_stocks['period_mean_distance']
                short_weights = short_stocks['period_mean_distance']
                
                # 标准化权重
                long_weights = long_weights / long_weights.sum()
                short_weights = short_weights / short_weights.sum()
                
                long_return = (long_stocks['nxt_ret'] * long_weights).sum()
                short_return = (short_stocks['nxt_ret'] * short_weights).sum()
                
                portfolio_return = long_return - short_return
                
                portfolio_returns.append({
                    'tradeDate': trade_date,
                    'portfolio_return': portfolio_return,
                    'long_return': long_return,
                    'short_return': short_return,
                    'n_stocks': n_stocks
                })
        
        portfolio_df = pd.DataFrame(portfolio_returns)
        
        if len(portfolio_df) > 0:
            portfolio_df = portfolio_df.sort_values('tradeDate')
            portfolio_df['cum_return'] = (1 + portfolio_df['portfolio_return']).cumprod()
            
            if strategy_type == 'enhanced_long_short':
                portfolio_df['cum_long'] = (1 + portfolio_df['long_return']).cumprod()
                portfolio_df['cum_short'] = (1 + portfolio_df['short_return']).cumprod()
                
                print(f"    平均多头收益: {portfolio_df['long_return'].mean():.4f}")
                print(f"    平均空头收益: {portfolio_df['short_return'].mean():.4f}")
                print(f"    平均组合收益: {portfolio_df['portfolio_return'].mean():.4f}")
                print(f"    杠杆倍数: {portfolio_df['leverage'].iloc[0]:.1f}x")
            else:
                print(f"    平均组合收益: {portfolio_df['portfolio_return'].mean():.4f}")
            
            print(f"    组合收益胜率: {(portfolio_df['portfolio_return'] > 0).mean():.2%}")
        
        return portfolio_df
        
    except Exception as e:
        print(f"  投资组合构建失败: {e}")
        return None

def calculate_enhanced_metrics(portfolio_returns, return_col='portfolio_return'):
    """计算增强版策略指标"""
    try:
        returns = portfolio_returns[return_col]
        cum_returns = (1 + returns).cumprod()
        
        # 基本统计
        total_return = cum_returns.iloc[-1] - 1
        n_periods = len(returns)
        annualized_return = (cum_returns.iloc[-1] ** (52 / n_periods)) - 1
        volatility = returns.std() * np.sqrt(52)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤和回撤期间
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # 找到最大回撤期间
        max_dd_end = drawdown.idxmin()
        max_dd_start = cum_returns[:max_dd_end].idxmax()
        
        # 胜率和盈亏比
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calmar比率 (年化收益率/最大回撤)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 信息比率 (假设基准收益为0)
        information_ratio = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
        
        metrics = {
            '总收益率': f"{total_return:.2%}",
            '年化收益率': f"{annualized_return:.2%}",
            '年化波动率': f"{volatility:.2%}",
            '夏普比率': f"{sharpe_ratio:.4f}",
            '信息比率': f"{information_ratio:.4f}",
            'Calmar比率': f"{calmar_ratio:.4f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '胜率': f"{win_rate:.2%}",
            '盈亏比': f"{profit_loss_ratio:.4f}",
            '交易次数': f"{len(returns)}次",
            '年化收益/波动': f"{annualized_return/volatility:.4f}" if volatility > 0 else "N/A"
        }
        
        return metrics
        
    except Exception as e:
        print(f"  策略指标计算失败: {e}")
        return {}

def run_enhanced_analysis():
    """运行增强版分析"""
    
    print("=" * 80)
    print("网络动量选股策略 - 增强版分析 (2015-2023)")
    print("=" * 80)
    
    # 1. 加载网络动量数据
    print("\n第一步: 加载网络动量数据...")
    try:
        network_momentum_df = pd.read_csv('网络动量_长期分析结果.csv')
        print(f"✓ 成功加载 {len(network_momentum_df)} 条网络动量记录")
        print(f"  原始日期范围: {network_momentum_df['tradeDate'].min()} 到 {network_momentum_df['tradeDate'].max()}")
    except Exception as e:
        print(f"✗ 加载网络动量数据失败: {e}")
        return None
    
    # 2. 获取扩展的收益率数据 (2015-2023)
    print("\n第二步: 获取扩展收益率数据 (2015-2023)...")
    start_time = time.time()
    
    # 扩展到2023年
    years = [
        ('2015-01-01', '2015-12-31'), ('2016-01-01', '2016-12-31'), 
        ('2017-01-01', '2017-12-31'), ('2018-01-01', '2018-12-31'), 
        ('2019-01-01', '2019-12-31'), ('2020-01-01', '2020-12-31'),
        ('2021-01-01', '2021-12-31'), ('2022-01-01', '2022-12-31'),
        ('2023-01-01', '2023-12-31')
    ]
    
    bt_mret_dfs = []
    for start, end in years:
        try:
            year_data = DataAPI.MktEquwAdjGet(beginDate=start, endDate=end, field="secID,endDate,chgPct")
            if len(year_data) > 0:
                bt_mret_dfs.append(year_data)
                print(f"    {start[:4]}年: {len(year_data)} 条记录")
            else:
                print(f"    {start[:4]}年: 无数据")
        except Exception as e:
            print(f"    {start[:4]}年: 获取失败 - {e}")
    
    if not bt_mret_dfs:
        print("✗ 没有获取到收益率数据")
        return None
    
    bt_mret_df = pd.concat(bt_mret_dfs, ignore_index=True)
    
    # 处理收益率数据
    bt_mret_df.rename(columns={'endDate':'tradeDate', 'chgPct':'curr_ret'}, inplace=True)
    bt_mret_df['ticker'] = bt_mret_df['secID'].str.slice(0,6)
    bt_mret_df['tradeDate'] = pd.to_datetime(bt_mret_df['tradeDate']).dt.strftime('%Y-%m-%d')
    bt_mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    bt_mret_df['nxt_ret'] = bt_mret_df.groupby('ticker')['curr_ret'].shift(-1)
    bt_mret_df = bt_mret_df.dropna(subset=['nxt_ret'])
    
    print(f"✓ 获取到 {len(bt_mret_df)} 条收益率数据")
    print(f"  收益率日期范围: {bt_mret_df['tradeDate'].min()} 到 {bt_mret_df['tradeDate'].max()}")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    return network_momentum_df, bt_mret_df

if __name__ == "__main__":
    try:
        print("开始增强版网络动量分析...")
        result = run_enhanced_analysis()

        if result:
            network_momentum_df, bt_mret_df = result
            print("\n✓ 数据准备完成！")

            # 3. 合并数据并运行增强策略
            print("\n第三步: 合并数据...")
            back_test_date = '2015-06-01'
            network_momentum_filtered = network_momentum_df[
                network_momentum_df['tradeDate'] >= back_test_date
            ].copy()

            # 合并因子和收益率数据
            factor_return_df = pd.merge(
                network_momentum_filtered,
                bt_mret_df[['secID', 'tradeDate', 'nxt_ret']],
                on=['secID', 'tradeDate'],
                how='inner'
            )

            print(f"✓ 合并完成，得到 {len(factor_return_df)} 条有效记录")
            print(f"  回测期间: {factor_return_df['tradeDate'].min()} 到 {factor_return_df['tradeDate'].max()}")
            print(f"  股票数量: {factor_return_df['secID'].nunique()}")

            if len(factor_return_df) == 0:
                print("✗ 没有匹配的数据")
                exit()

            # 4. 运行多种增强策略
            print("\n第四步: 运行增强策略...")

            strategies = [
                ('enhanced_long_short', '增强多空策略 (10%选股 + 2倍杠杆)'),
                ('momentum_only', '纯多头动量策略'),
                ('market_neutral', '市场中性策略')
            ]

            strategy_results = {}

            for strategy_type, strategy_name in strategies:
                print(f"\n--- {strategy_name} ---")
                start_time = time.time()

                portfolio_returns = calculate_enhanced_portfolio_returns(factor_return_df, strategy_type)

                if portfolio_returns is not None and len(portfolio_returns) > 0:
                    print(f"✓ {strategy_name}构建完成，得到 {len(portfolio_returns)} 期收益")

                    # 计算策略指标
                    metrics = calculate_enhanced_metrics(portfolio_returns, 'portfolio_return')
                    strategy_results[strategy_type] = {
                        'returns': portfolio_returns,
                        'metrics': metrics
                    }

                    print(f"  年化收益率: {metrics.get('年化收益率', 'N/A')}")
                    print(f"  夏普比率: {metrics.get('夏普比率', 'N/A')}")
                    print(f"  最大回撤: {metrics.get('最大回撤', 'N/A')}")
                    print(f"  处理时间: {time.time() - start_time:.2f}秒")
                else:
                    print(f"✗ {strategy_name}构建失败")

            # 5. 策略对比分析
            print("\n第五步: 策略对比分析...")
            print("\n" + "=" * 100)
            print(f"{'策略名称':<25} {'年化收益率':<12} {'夏普比率':<10} {'最大回撤':<10} {'胜率':<8} {'Calmar比率':<12}")
            print("=" * 100)

            for strategy_type, strategy_name in strategies:
                if strategy_type in strategy_results:
                    metrics = strategy_results[strategy_type]['metrics']
                    print(f"{strategy_name:<25} {metrics.get('年化收益率', 'N/A'):<12} "
                          f"{metrics.get('夏普比率', 'N/A'):<10} {metrics.get('最大回撤', 'N/A'):<10} "
                          f"{metrics.get('胜率', 'N/A'):<8} {metrics.get('Calmar比率', 'N/A'):<12}")

            # 6. 保存结果
            print("\n第六步: 保存结果...")
            for strategy_type in strategy_results:
                filename = f'增强策略_{strategy_type}_结果.csv'
                strategy_results[strategy_type]['returns'].to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"✓ {filename} 已保存")

            print("\n" + "=" * 80)
            print("增强版分析完成！")
            print("=" * 80)

            # 推荐最佳策略
            best_strategy = None
            best_sharpe = -999

            for strategy_type in strategy_results:
                sharpe_str = strategy_results[strategy_type]['metrics'].get('夏普比率', '0.0000')
                try:
                    sharpe_val = float(sharpe_str)
                    if sharpe_val > best_sharpe:
                        best_sharpe = sharpe_val
                        best_strategy = strategy_type
                except:
                    pass

            if best_strategy:
                strategy_name = dict(strategies)[best_strategy]
                metrics = strategy_results[best_strategy]['metrics']
                print(f"\n🏆 推荐策略: {strategy_name}")
                print(f"   年化收益率: {metrics.get('年化收益率', 'N/A')}")
                print(f"   夏普比率: {metrics.get('夏普比率', 'N/A')}")
                print(f"   最大回撤: {metrics.get('最大回撤', 'N/A')}")

            print("\n✓ 增强版分析执行成功！")

        else:
            print("\n✗ 数据准备失败！")

    except Exception as e:
        print(f"\n✗ 增强版分析出错: {e}")
        import traceback
        traceback.print_exc()
