# -*- coding: utf-8 -*-
"""
网络动量选股策略 - 长期回测分析
基于已生成的长期网络动量数据进行完整回测
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

def calculate_portfolio_returns(factor_return_df, method='long_short'):
    """
    基于网络动量因子构建投资组合并计算收益率
    
    Args:
        factor_return_df: 包含因子和收益率的DataFrame
        method: 投资组合方法 ('long_short', 'long_only', 'quintile')
        
    Returns:
        portfolio_returns: 投资组合收益率时间序列
    """
    try:
        print(f"  构建{method}投资组合...")
        
        portfolio_returns = []
        
        # 按日期分组进行投资组合构建
        for trade_date, group in factor_return_df.groupby('tradeDate'):
            group = group.dropna(subset=['period_mean_distance', 'nxt_ret'])
            
            if len(group) < 50:  # 至少需要50只股票
                continue
            
            # 按网络动量因子排序
            group = group.sort_values('period_mean_distance')
            
            if method == 'long_short':
                # 多空策略：选择前20%和后20%的股票
                n_stocks = len(group)
                n_select = max(20, n_stocks // 5)
                
                # 多头：网络距离最小的股票（更相似，可能表现更好）
                long_stocks = group.head(n_select)
                # 空头：网络距离最大的股票（更孤立，可能表现更差）
                short_stocks = group.tail(n_select)
                
                long_return = long_stocks['nxt_ret'].mean()
                short_return = short_stocks['nxt_ret'].mean()
                portfolio_return = long_return - short_return
                
                portfolio_returns.append({
                    'tradeDate': trade_date,
                    'portfolio_return': portfolio_return,
                    'long_return': long_return,
                    'short_return': short_return,
                    'n_stocks': n_stocks,
                    'n_long': len(long_stocks),
                    'n_short': len(short_stocks)
                })
                
            elif method == 'quintile':
                # 五分位数策略
                n_stocks = len(group)
                quintile_size = n_stocks // 5
                
                quintile_returns = []
                for i in range(5):
                    start_idx = i * quintile_size
                    end_idx = (i + 1) * quintile_size if i < 4 else n_stocks
                    quintile_stocks = group.iloc[start_idx:end_idx]
                    quintile_return = quintile_stocks['nxt_ret'].mean()
                    quintile_returns.append(quintile_return)
                
                portfolio_returns.append({
                    'tradeDate': trade_date,
                    'Q1_return': quintile_returns[0],  # 最小距离
                    'Q2_return': quintile_returns[1],
                    'Q3_return': quintile_returns[2],
                    'Q4_return': quintile_returns[3],
                    'Q5_return': quintile_returns[4],  # 最大距离
                    'Q1_Q5_spread': quintile_returns[0] - quintile_returns[4],
                    'n_stocks': n_stocks
                })
        
        portfolio_df = pd.DataFrame(portfolio_returns)
        
        if len(portfolio_df) > 0:
            portfolio_df = portfolio_df.sort_values('tradeDate')
            
            if method == 'long_short':
                # 计算累计收益率
                portfolio_df['cum_return'] = (1 + portfolio_df['portfolio_return']).cumprod()
                portfolio_df['cum_long'] = (1 + portfolio_df['long_return']).cumprod()
                portfolio_df['cum_short'] = (1 + portfolio_df['short_return']).cumprod()
                
                print(f"    平均多头收益: {portfolio_df['long_return'].mean():.4f}")
                print(f"    平均空头收益: {portfolio_df['short_return'].mean():.4f}")
                print(f"    平均多空收益: {portfolio_df['portfolio_return'].mean():.4f}")
                print(f"    多空收益胜率: {(portfolio_df['portfolio_return'] > 0).mean():.2%}")
                
            elif method == 'quintile':
                # 计算五分位数累计收益率
                for i in range(1, 6):
                    portfolio_df[f'cum_Q{i}'] = (1 + portfolio_df[f'Q{i}_return']).cumprod()
                portfolio_df['cum_spread'] = (1 + portfolio_df['Q1_Q5_spread']).cumprod()
                
                print(f"    Q1平均收益: {portfolio_df['Q1_return'].mean():.4f}")
                print(f"    Q5平均收益: {portfolio_df['Q5_return'].mean():.4f}")
                print(f"    Q1-Q5价差: {portfolio_df['Q1_Q5_spread'].mean():.4f}")
                print(f"    价差胜率: {(portfolio_df['Q1_Q5_spread'] > 0).mean():.2%}")
        
        return portfolio_df
        
    except Exception as e:
        print(f"  投资组合构建失败: {e}")
        return None

def calculate_strategy_metrics(portfolio_returns, return_col='portfolio_return'):
    """计算策略表现指标"""
    try:
        returns = portfolio_returns[return_col]
        cum_returns = (1 + returns).cumprod()
        
        # 基本统计
        total_return = cum_returns.iloc[-1] - 1
        n_periods = len(returns)
        annualized_return = (cum_returns.iloc[-1] ** (52 / n_periods)) - 1  # 假设周度数据
        volatility = returns.std() * np.sqrt(52)
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

def plot_long_term_equity_curve(portfolio_returns, quintile_returns=None):
    """绘制长期权益曲线"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 多空策略权益曲线
        ax1 = axes[0, 0]
        ax1.plot(portfolio_returns['tradeDate'], portfolio_returns['cum_return'], 
                label='多空组合', linewidth=2, color='red')
        ax1.plot(portfolio_returns['tradeDate'], portfolio_returns['cum_long'], 
                label='多头组合', linewidth=1, color='green', alpha=0.7)
        ax1.plot(portfolio_returns['tradeDate'], portfolio_returns['cum_short'], 
                label='空头组合', linewidth=1, color='blue', alpha=0.7)
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('多空策略权益曲线 (2015-2019)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('累计收益率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收益率分布
        ax2 = axes[0, 1]
        ax2.hist(portfolio_returns['portfolio_return'], bins=50, alpha=0.7, color='purple')
        ax2.axvline(x=portfolio_returns['portfolio_return'].mean(), color='red', 
                   linestyle='--', label=f'均值: {portfolio_returns["portfolio_return"].mean():.4f}')
        ax2.set_title('多空收益率分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('周度收益率')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 五分位数策略
        if quintile_returns is not None:
            ax3 = axes[1, 0]
            for i in range(1, 6):
                ax3.plot(quintile_returns['tradeDate'], quintile_returns[f'cum_Q{i}'], 
                        label=f'Q{i}', linewidth=1)
            ax3.plot(quintile_returns['tradeDate'], quintile_returns['cum_spread'], 
                    label='Q1-Q5价差', linewidth=2, color='red')
            ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('五分位数策略权益曲线', fontsize=12, fontweight='bold')
            ax3.set_ylabel('累计收益率')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 滚动夏普比率
        ax4 = axes[1, 1]
        rolling_window = 12  # 12周滚动
        rolling_returns = portfolio_returns['portfolio_return'].rolling(rolling_window)
        rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(52)
        ax4.plot(portfolio_returns['tradeDate'], rolling_sharpe, color='orange', linewidth=1)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(y=rolling_sharpe.mean(), color='red', linestyle='--', 
                   label=f'平均: {rolling_sharpe.mean():.2f}')
        ax4.set_title('滚动夏普比率 (12周)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('夏普比率')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('网络动量策略_长期回测结果.png', dpi=300, bbox_inches='tight')
        print("  ✓ 长期回测图表已保存为: 网络动量策略_长期回测结果.png")
        
        plt.show()
        
    except Exception as e:
        print(f"  图表绘制失败: {e}")

def run_long_term_backtest():
    """运行长期回测分析"""
    
    print("=" * 80)
    print("网络动量选股策略 - 长期回测分析 (2015-2019)")
    print("=" * 80)
    
    # 1. 加载长期网络动量数据
    print("\n第一步: 加载网络动量数据...")
    try:
        network_momentum_df = pd.read_csv('网络动量_长期分析结果.csv')
        print(f"✓ 成功加载 {len(network_momentum_df)} 条网络动量记录")
        print(f"  日期范围: {network_momentum_df['tradeDate'].min()} 到 {network_momentum_df['tradeDate'].max()}")
        print(f"  股票数量: {network_momentum_df['secID'].nunique()}")
    except Exception as e:
        print(f"✗ 加载网络动量数据失败: {e}")
        return None
    
    # 2. 获取收益率数据
    print("\n第二步: 获取收益率数据...")
    start_time = time.time()
    
    start_date = "2015-01-01"
    end_date = "2019-08-23"
    
    # 分段获取收益率数据
    print("  分段获取收益率数据...")
    bt_mret_dfs = []
    years = [('2015-01-01', '2015-12-31'), ('2016-01-01', '2016-12-31'), 
             ('2017-01-01', '2017-12-31'), ('2018-01-01', '2018-12-31'), 
             ('2019-01-01', '2019-08-23')]
    
    for start, end in years:
        year_data = DataAPI.MktEquwAdjGet(beginDate=start, endDate=end, field="secID,endDate,chgPct")
        bt_mret_dfs.append(year_data)
        print(f"    {start[:4]}年: {len(year_data)} 条记录")
    
    bt_mret_df = pd.concat(bt_mret_dfs, ignore_index=True)
    
    # 处理收益率数据
    bt_mret_df.rename(columns={'endDate':'tradeDate', 'chgPct':'curr_ret'}, inplace=True)
    bt_mret_df['ticker'] = bt_mret_df['secID'].str.slice(0,6)

    # 确保日期格式一致
    bt_mret_df['tradeDate'] = pd.to_datetime(bt_mret_df['tradeDate']).dt.strftime('%Y-%m-%d')

    bt_mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
    bt_mret_df['nxt_ret'] = bt_mret_df.groupby('ticker')['curr_ret'].shift(-1)
    bt_mret_df = bt_mret_df.dropna(subset=['nxt_ret'])
    
    print(f"✓ 获取到 {len(bt_mret_df)} 条收益率数据")
    print(f"  处理时间: {time.time() - start_time:.2f}秒")
    
    return network_momentum_df, bt_mret_df

if __name__ == "__main__":
    try:
        print("开始长期回测分析...")
        result = run_long_term_backtest()

        if result:
            network_momentum_df, bt_mret_df = result
            print("\n✓ 数据准备完成，开始回测分析...")

            # 3. 合并因子和收益率数据
            print("\n第三步: 合并因子和收益率数据...")
            start_time = time.time()

            # 过滤回测期间的网络动量数据
            back_test_date = '2015-06-01'
            network_momentum_filtered = network_momentum_df[
                network_momentum_df['tradeDate'] >= back_test_date
            ].copy()

            print(f"  过滤后网络动量数据: {len(network_momentum_filtered)} 条")
            print(f"  回测开始日期: {back_test_date}")

            # 合并数据
            factor_return_df = pd.merge(
                network_momentum_filtered,
                bt_mret_df[['secID', 'tradeDate', 'nxt_ret']],
                on=['secID', 'tradeDate'],
                how='inner'
            )

            print(f"✓ 合并完成，得到 {len(factor_return_df)} 条有效记录")
            print(f"  股票数量: {factor_return_df['secID'].nunique()}")
            print(f"  日期数量: {factor_return_df['tradeDate'].nunique()}")
            print(f"  处理时间: {time.time() - start_time:.2f}秒")

            if len(factor_return_df) == 0:
                print("✗ 没有匹配的因子和收益率数据")
                exit()

            # 4. 计算IC分析
            print("\n第四步: IC分析...")
            start_time = time.time()

            period_ic = factor_return_df.groupby('tradeDate').apply(
                lambda x: x['period_mean_distance'].corr(x['nxt_ret'])
            ).dropna()

            print(f"✓ IC分析完成，有效期数: {len(period_ic)}")

            # IC统计
            ic_mean = period_ic.mean()
            ic_std = period_ic.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(period_ic))) if ic_std > 0 else 0

            print(f"\nIC统计结果:")
            print(f"  IC均值: {ic_mean:.4f}")
            print(f"  IC标准差: {ic_std:.4f}")
            print(f"  ICIR: {icir:.4f}")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  IC>0比例: {(period_ic > 0).mean():.2%}")
            print(f"  最大IC: {period_ic.max():.4f}")
            print(f"  最小IC: {period_ic.min():.4f}")
            print(f"  处理时间: {time.time() - start_time:.2f}秒")

            # 5. 构建投资组合
            print("\n第五步: 构建投资组合...")
            start_time = time.time()

            # 多空策略
            portfolio_returns = calculate_portfolio_returns(factor_return_df, 'long_short')

            if portfolio_returns is not None and len(portfolio_returns) > 0:
                print(f"✓ 多空组合构建完成，得到 {len(portfolio_returns)} 期收益")

                # 五分位数策略
                quintile_returns = calculate_portfolio_returns(factor_return_df, 'quintile')

                if quintile_returns is not None:
                    print(f"✓ 五分位数组合构建完成，得到 {len(quintile_returns)} 期收益")

                print(f"  处理时间: {time.time() - start_time:.2f}秒")

                # 6. 计算策略表现指标
                print("\n第六步: 策略表现分析...")

                # 多空策略指标
                print("\n多空策略表现指标:")
                ls_metrics = calculate_strategy_metrics(portfolio_returns, 'portfolio_return')
                for metric, value in ls_metrics.items():
                    print(f"  {metric}: {value}")

                # 五分位数策略指标
                if quintile_returns is not None:
                    print("\n五分位数策略表现指标:")
                    q_metrics = calculate_strategy_metrics(quintile_returns, 'Q1_Q5_spread')
                    for metric, value in q_metrics.items():
                        print(f"  {metric}: {value}")

                # 7. 绘制权益曲线
                print("\n第七步: 生成权益曲线...")
                plot_long_term_equity_curve(portfolio_returns, quintile_returns)

                # 8. 保存结果
                print("\n第八步: 保存结果...")
                portfolio_returns.to_csv('长期回测_多空策略结果.csv', index=False, encoding='utf-8-sig')
                if quintile_returns is not None:
                    quintile_returns.to_csv('长期回测_五分位数策略结果.csv', index=False, encoding='utf-8-sig')

                # 保存IC分析结果
                ic_results = pd.DataFrame({
                    'tradeDate': period_ic.index,
                    'IC': period_ic.values
                })
                ic_results.to_csv('长期回测_IC分析结果.csv', index=False, encoding='utf-8-sig')

                print("✓ 结果已保存到CSV文件")

                print("\n" + "=" * 80)
                print("长期回测分析完成！")
                print("=" * 80)
                print(f"回测期间: {back_test_date} 到 {factor_return_df['tradeDate'].max()}")
                print(f"总交易期数: {len(portfolio_returns)}")
                print(f"年化收益率: {ls_metrics.get('年化收益率', 'N/A')}")
                print(f"夏普比率: {ls_metrics.get('夏普比率', 'N/A')}")
                print(f"最大回撤: {ls_metrics.get('最大回撤', 'N/A')}")
                print(f"胜率: {ls_metrics.get('胜率', 'N/A')}")
                print("\n✓ 长期回测执行成功！")

            else:
                print("✗ 投资组合构建失败")
        else:
            print("\n✗ 数据准备失败！")

    except Exception as e:
        print(f"\n✗ 长期回测分析出错: {e}")
        import traceback
        traceback.print_exc()
