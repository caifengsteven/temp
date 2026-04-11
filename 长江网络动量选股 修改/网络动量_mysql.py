# -*- coding: utf-8 -*-
"""
基于MySQL数据库的网络动量选股策略
修改自原始的网络动量.py，使用本地MySQL数据库替代外部API
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import time
import os
import json
from datetime import datetime, timedelta
import logging

# 导入自定义模块
from database_setup import DatabaseManager, setup_database
from data_manager import DataManager

# 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkMomentumStrategy:
    def __init__(self, config_file='para.json'):
        """
        初始化网络动量策略
        
        Args:
            config_file (str): 配置文件路径
        """
        self.config_file = config_file
        self.data_manager = DataManager(config_file)
        self.factor_list = ['mass_index', 'kdj_j', 'rsi', 'cci10', 'cmo', 'mfi']
        
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        logger.info("网络动量策略初始化完成")
    
    def setup_database_if_needed(self):
        """如果需要，设置数据库"""
        try:
            # 测试数据库连接
            test_df = self.data_manager.get_trade_calendar('2020-01-01', '2020-01-31')
            if len(test_df) == 0:
                logger.info("数据库为空，需要初始化...")
                return False
            return True
        except Exception as e:
            logger.warning(f"数据库连接测试失败: {e}")
            logger.info("正在设置数据库...")
            db_manager = setup_database()
            if db_manager:
                db_manager.close_connection()
                return True
            return False
    
    def load_sample_data(self):
        """
        加载示例数据到数据库（用于演示）
        在实际使用中，您需要从真实数据源导入数据
        """
        logger.info("正在生成示例数据...")
        
        # 生成示例交易日历
        start_date = pd.Timestamp('2012-01-01')
        end_date = pd.Timestamp('2019-08-23')
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        calendar_data = []
        for date in date_range:
            # 简单的交易日判断（排除周末）
            is_open = 1 if date.weekday() < 5 else 0
            is_week_end = 1 if date.weekday() == 4 and is_open else 0  # 周五
            is_month_end = 1 if date == date + pd.offsets.MonthEnd(0) and is_open else 0
            
            calendar_data.append({
                'calendar_date': date,
                'is_open': is_open,
                'is_week_end': is_week_end,
                'is_month_end': is_month_end,
                'exchange_cd': 'XSHG'
            })
        
        calendar_df = pd.DataFrame(calendar_data)
        self.data_manager.save_trade_calendar(calendar_df)
        
        # 生成示例股票因子数据
        stock_codes = [f"00000{i}.XSHE" for i in range(1, 101)]  # 100只示例股票
        trade_dates = calendar_df[calendar_df['is_open'] == 1]['calendar_date'].values
        
        factor_data = []
        for stock_code in stock_codes:
            ticker = stock_code[:6]
            for trade_date in trade_dates[::5]:  # 每5个交易日一个数据点
                # 生成随机因子数据
                factor_data.append({
                    'sec_id': stock_code,
                    'ticker': ticker,
                    'trade_date': trade_date,
                    'mass_index': np.random.normal(1.0, 0.2),
                    'kdj_j': np.random.normal(50, 20),
                    'rsi': np.random.normal(50, 15),
                    'cci10': np.random.normal(0, 100),
                    'cmo': np.random.normal(0, 20),
                    'mfi': np.random.normal(50, 20),
                    'lcap': np.random.normal(20, 2)  # 对数市值
                })
        
        factor_df = pd.DataFrame(factor_data)
        self.data_manager.save_stock_factors(factor_df)
        
        # 生成示例收益率数据
        returns_data = []
        for stock_code in stock_codes:
            ticker = stock_code[:6]
            for trade_date in trade_dates[::5]:
                returns_data.append({
                    'sec_id': stock_code,
                    'ticker': ticker,
                    'end_date': trade_date,
                    'chg_pct': np.random.normal(0, 0.02),  # 日收益率
                    'close_price': np.random.uniform(10, 100),
                    'volume': np.random.randint(1000000, 10000000),
                    'turnover': np.random.uniform(1000000, 100000000)
                })
        
        returns_df = pd.DataFrame(returns_data)
        self.data_manager.save_stock_returns(returns_df)
        
        logger.info("示例数据生成完成")
    
    def get_trade_calendar(self, start_date, end_date):
        """获取交易日历"""
        calendar_df = self.data_manager.get_trade_calendar(start_date, end_date)
        
        # 转换列名以匹配原始代码
        calendar_df = calendar_df.rename(columns={
            'calendar_date': 'calendarDate',
            'is_open': 'isOpen',
            'is_week_end': 'isWeekEnd',
            'is_month_end': 'isMonthEnd'
        })
        
        return calendar_df
    
    def get_factor_data(self, week_end_list):
        """获取因子数据"""
        # 将日期列表转换为字符串
        date_strings = [str(date) for date in week_end_list]
        
        factor_df_list = []
        for date_str in date_strings:
            factor_data = self.data_manager.get_stock_factors(date_str, date_str)
            if len(factor_data) > 0:
                # 转换列名以匹配原始代码
                factor_data = factor_data.rename(columns={
                    'sec_id': 'secID',
                    'trade_date': 'tradeDate',
                    'mass_index': 'MassIndex',
                    'kdj_j': 'KDJ_J',
                    'rsi': 'RSI',
                    'cci10': 'CCI10',
                    'cmo': 'CMO',
                    'mfi': 'MFI'
                })
                factor_df_list.append(factor_data)
        
        if factor_df_list:
            factor_df = pd.concat(factor_df_list, axis=0)
            return factor_df
        else:
            return pd.DataFrame()
    
    def get_returns_data(self, start_date, end_date):
        """获取收益率数据"""
        returns_df = self.data_manager.get_stock_returns(start_date, end_date)
        
        # 转换列名以匹配原始代码
        returns_df = returns_df.rename(columns={
            'sec_id': 'secID',
            'end_date': 'endDate',
            'chg_pct': 'chgPct'
        })
        
        return returns_df
    
    def cal_std_factor(self, df, step=52):
        """对指标值进行时间序列标准化"""
        factor_list = ['MassIndex', 'KDJ_J', 'RSI', 'CCI10', 'CMO', 'MFI']
        df = df.copy()
        df = df.sort_values('tradeDate')
        for f in factor_list:
            if f in df.columns:
                df[f] = (df[f] - df[f].rolling(step, min_periods=30).mean()) / df[f].rolling(step, min_periods=30).std()
        return df
    
    def cal_distance(self, df):
        """计算每个节点与其他节点的平均距离"""
        df1 = df.copy()
        df1 = df1.set_index('secID').drop(['ticker', 'tradeDate'], axis=1).sort_index()
        df2 = df1.values.repeat([len(df1)] * len(df1), axis=0)
        df3 = np.concatenate([df1] * len(df1))
        df4 = pd.DataFrame(np.sqrt(np.sum((df2 - df3) ** 2, axis=1)).reshape(len(df1), -1), 
                          index=df1.index, columns=df1.index)
        df5 = df4.mean(axis=1)
        del df1, df2, df3, df4
        return df5
    
    def cal_period_mean_distance(self, df, period_step=4):
        """计算过去四周网络距离的平均值"""
        df = df.copy()
        df = df.sort_values('tradeDate')
        df['period_mean_distance'] = df['mean_distance'].rolling(period_step).mean()
        return df
    
    def run_strategy(self, start_date="2012-01-01", end_date="2019-08-23"):
        """运行网络动量策略"""
        logger.info("开始运行网络动量策略...")
        
        # 1. 检查并设置数据库
        if not self.setup_database_if_needed():
            logger.info("正在加载示例数据...")
            self.load_sample_data()
        
        # 2. 获取交易日历
        calendar_df = self.get_trade_calendar(start_date, end_date)
        week_end_list = calendar_df[calendar_df['isWeekEnd'] == 1]['calendarDate'].values
        
        logger.info(f"获取到 {len(week_end_list)} 个周末交易日")
        
        # 3. 获取因子数据
        factor_df = self.get_factor_data(week_end_list)
        
        if len(factor_df) == 0:
            logger.error("未获取到因子数据")
            return None
        
        logger.info(f"获取到 {len(factor_df)} 条因子数据")
        
        # 4. 对指标值进行时间序列标准化
        factor_df1 = factor_df.groupby(['secID', 'ticker'], as_index=False).apply(
            lambda x: self.cal_std_factor(x, 52)
        ).reset_index(drop=True)
        
        factor_df1 = factor_df1.dropna()
        factor_df1 = factor_df1[['secID', 'ticker', 'tradeDate', 'KDJ_J', 'RSI', 'CCI10', 'MFI', 'MassIndex', 'CMO']]
        
        logger.info(f"标准化后剩余 {len(factor_df1)} 条数据")
        
        # 5. 计算网络动量
        stock_distance_matrix = factor_df1.groupby('tradeDate').apply(lambda x: self.cal_distance(x))
        stock_distance_matrix = stock_distance_matrix.reset_index().rename(columns={0: 'mean_distance'})
        
        # 计算过去四周网络距离的平均值
        stock_distance_matrix1 = stock_distance_matrix.groupby('secID', as_index=False).apply(
            lambda x: self.cal_period_mean_distance(x, 4)
        ).dropna().reset_index(drop=True)
        
        logger.info(f"计算得到 {len(stock_distance_matrix1)} 条网络动量数据")
        
        # 6. 保存网络动量因子到数据库
        self.data_manager.save_network_momentum(stock_distance_matrix1)
        
        # 7. 获取收益率数据进行回测
        bt_mret_df = self.get_returns_data(start_date, end_date)
        
        if len(bt_mret_df) == 0:
            logger.error("未获取到收益率数据")
            return None
        
        # 处理收益率数据
        bt_mret_df.rename(columns={'endDate': 'tradeDate', 'chgPct': 'curr_ret'}, inplace=True)
        bt_mret_df['ticker'] = bt_mret_df['secID'].str.slice(0, 6)
        bt_mret_df.sort_values(['ticker', 'tradeDate'], inplace=True)
        bt_mret_df['nxt_ret'] = bt_mret_df.groupby('ticker')['curr_ret'].shift(-1)
        bt_mret_df = bt_mret_df.dropna(subset=['nxt_ret'])
        
        logger.info(f"处理后收益率数据 {len(bt_mret_df)} 条")
        
        # 8. 因子测试
        back_test_date = '2014-01-01'
        stock_distance_matrix2 = stock_distance_matrix1[stock_distance_matrix1['tradeDate'] >= back_test_date]
        
        factor_rtn_df = stock_distance_matrix2.merge(bt_mret_df, on=['secID', 'tradeDate'])
        
        if len(factor_rtn_df) == 0:
            logger.error("因子和收益率数据合并后为空")
            return None
        
        # 计算IC
        period_ic = factor_rtn_df.groupby('tradeDate').apply(
            lambda x: x[['period_mean_distance', 'nxt_ret']].corr(method="spearman").values[0, 1]
        )
        
        ic = period_ic.mean()
        std = period_ic.std()
        icir = ic / std if std != 0 else 0
        ic_t = stats.ttest_1samp(period_ic, 0)[0]
        
        ic_summary = pd.DataFrame([ic, std, icir, ic_t], 
                                 index=['IC均值', 'IC波动率', 'ICIR', 't值'], 
                                 columns=['网络动量因子']).T.applymap(lambda x: round(x, 3))
        
        logger.info("策略运行完成")
        logger.info(f"IC统计:\n{ic_summary}")
        
        return {
            'factor_data': stock_distance_matrix1,
            'ic_summary': ic_summary,
            'factor_return_data': factor_rtn_df,
            'period_ic': period_ic
        }
    
    def close(self):
        """关闭数据库连接"""
        self.data_manager.close()

def main():
    """主函数"""
    strategy = NetworkMomentumStrategy()
    
    try:
        results = strategy.run_strategy()
        if results:
            print("策略运行成功！")
            print("\nIC统计结果:")
            print(results['ic_summary'])
        else:
            print("策略运行失败！")
    except Exception as e:
        logger.error(f"策略运行出错: {e}")
    finally:
        strategy.close()

if __name__ == "__main__":
    main()
