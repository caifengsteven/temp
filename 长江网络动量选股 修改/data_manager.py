# -*- coding: utf-8 -*-
"""
数据管理模块
用于网络动量选股策略的数据获取、存储和检索
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from database_setup import DatabaseManager

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config_file='para.json'):
        """
        初始化数据管理器
        
        Args:
            config_file (str): 配置文件路径
        """
        self.db_manager = DatabaseManager(config_file)
        self.db_manager.connect()
        self.db_manager.mysql_config['database'] = 'network_momentum'
        
        # 重新连接到指定数据库
        try:
            import mysql.connector
            self.db_manager.connection = mysql.connector.connect(
                host=self.db_manager.mysql_config['host'],
                user=self.db_manager.mysql_config['user_name'],
                password=self.db_manager.mysql_config['pass_wd'],
                port=self.db_manager.mysql_config['port'],
                database=self.db_manager.mysql_config['database'],
                auth_plugin='mysql_native_password'
            )
            logger.info("数据管理器初始化成功")
        except Exception as e:
            logger.error(f"数据管理器初始化失败: {e}")
    
    def save_trade_calendar(self, calendar_df):
        """
        保存交易日历数据
        
        Args:
            calendar_df (pd.DataFrame): 交易日历数据
        """
        try:
            # 重命名列以匹配数据库表结构
            calendar_df_clean = calendar_df.copy()
            column_mapping = {
                'calendarDate': 'calendar_date',
                'isOpen': 'is_open',
                'isWeekEnd': 'is_week_end',
                'isMonthEnd': 'is_month_end',
                'exchangeCD': 'exchange_cd'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in calendar_df_clean.columns:
                    calendar_df_clean = calendar_df_clean.rename(columns={old_col: new_col})
            
            # 确保日期格式正确
            calendar_df_clean['calendar_date'] = pd.to_datetime(calendar_df_clean['calendar_date'])
            
            # 保存到数据库
            success = self.db_manager.insert_dataframe(calendar_df_clean, 'trade_calendar', if_exists='replace')
            if success:
                logger.info(f"成功保存 {len(calendar_df_clean)} 条交易日历记录")
            return success
        except Exception as e:
            logger.error(f"保存交易日历数据失败: {e}")
            return False
    
    def save_stock_factors(self, factor_df):
        """
        保存股票因子数据
        
        Args:
            factor_df (pd.DataFrame): 股票因子数据
        """
        try:
            factor_df_clean = factor_df.copy()
            
            # 重命名列以匹配数据库表结构
            column_mapping = {
                'secID': 'sec_id',
                'tradeDate': 'trade_date',
                'MassIndex': 'mass_index',
                'KDJ_J': 'kdj_j',
                'RSI': 'rsi',
                'CCI10': 'cci10',
                'CMO': 'cmo',
                'MFI': 'mfi',
                'LCAP': 'lcap'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in factor_df_clean.columns:
                    factor_df_clean = factor_df_clean.rename(columns={old_col: new_col})
            
            # 确保日期格式正确
            factor_df_clean['trade_date'] = pd.to_datetime(factor_df_clean['trade_date'])
            
            # 保存到数据库
            success = self.db_manager.insert_dataframe(factor_df_clean, 'stock_factors', if_exists='append')
            if success:
                logger.info(f"成功保存 {len(factor_df_clean)} 条股票因子记录")
            return success
        except Exception as e:
            logger.error(f"保存股票因子数据失败: {e}")
            return False
    
    def save_stock_returns(self, returns_df):
        """
        保存股票收益率数据
        
        Args:
            returns_df (pd.DataFrame): 股票收益率数据
        """
        try:
            returns_df_clean = returns_df.copy()
            
            # 重命名列以匹配数据库表结构
            column_mapping = {
                'secID': 'sec_id',
                'endDate': 'end_date',
                'chgPct': 'chg_pct',
                'closePrice': 'close_price'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in returns_df_clean.columns:
                    returns_df_clean = returns_df_clean.rename(columns={old_col: new_col})
            
            # 确保日期格式正确
            returns_df_clean['end_date'] = pd.to_datetime(returns_df_clean['end_date'])
            
            # 保存到数据库
            success = self.db_manager.insert_dataframe(returns_df_clean, 'stock_returns', if_exists='append')
            if success:
                logger.info(f"成功保存 {len(returns_df_clean)} 条股票收益率记录")
            return success
        except Exception as e:
            logger.error(f"保存股票收益率数据失败: {e}")
            return False
    
    def save_network_momentum(self, momentum_df):
        """
        保存网络动量因子数据
        
        Args:
            momentum_df (pd.DataFrame): 网络动量因子数据
        """
        try:
            momentum_df_clean = momentum_df.copy()
            
            # 重命名列以匹配数据库表结构
            column_mapping = {
                'secID': 'sec_id',
                'tradeDate': 'trade_date'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in momentum_df_clean.columns:
                    momentum_df_clean = momentum_df_clean.rename(columns={old_col: new_col})
            
            # 确保日期格式正确
            momentum_df_clean['trade_date'] = pd.to_datetime(momentum_df_clean['trade_date'])
            
            # 保存到数据库
            success = self.db_manager.insert_dataframe(momentum_df_clean, 'network_momentum', if_exists='append')
            if success:
                logger.info(f"成功保存 {len(momentum_df_clean)} 条网络动量因子记录")
            return success
        except Exception as e:
            logger.error(f"保存网络动量因子数据失败: {e}")
            return False
    
    def get_trade_calendar(self, start_date=None, end_date=None, exchange_cd='XSHG'):
        """
        获取交易日历数据
        
        Args:
            start_date (str): 开始日期
            end_date (str): 结束日期
            exchange_cd (str): 交易所代码
            
        Returns:
            pd.DataFrame: 交易日历数据
        """
        sql = """
        SELECT calendar_date, is_open, is_week_end, is_month_end, exchange_cd
        FROM trade_calendar
        WHERE exchange_cd = %s
        """
        params = [exchange_cd]
        
        if start_date:
            sql += " AND calendar_date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND calendar_date <= %s"
            params.append(end_date)
        
        sql += " ORDER BY calendar_date"
        
        return self.db_manager.query_dataframe(sql, params)
    
    def get_stock_factors(self, start_date=None, end_date=None, sec_ids=None):
        """
        获取股票因子数据
        
        Args:
            start_date (str): 开始日期
            end_date (str): 结束日期
            sec_ids (list): 股票代码列表
            
        Returns:
            pd.DataFrame: 股票因子数据
        """
        sql = """
        SELECT sec_id, ticker, trade_date, mass_index, kdj_j, rsi, cci10, cmo, mfi, lcap
        FROM stock_factors
        WHERE 1=1
        """
        params = []
        
        if start_date:
            sql += " AND trade_date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= %s"
            params.append(end_date)
        
        if sec_ids:
            placeholders = ','.join(['%s'] * len(sec_ids))
            sql += f" AND sec_id IN ({placeholders})"
            params.extend(sec_ids)
        
        sql += " ORDER BY trade_date, sec_id"
        
        return self.db_manager.query_dataframe(sql, params)
    
    def get_stock_returns(self, start_date=None, end_date=None, sec_ids=None):
        """
        获取股票收益率数据
        
        Args:
            start_date (str): 开始日期
            end_date (str): 结束日期
            sec_ids (list): 股票代码列表
            
        Returns:
            pd.DataFrame: 股票收益率数据
        """
        sql = """
        SELECT sec_id, ticker, end_date, chg_pct, close_price, volume, turnover
        FROM stock_returns
        WHERE 1=1
        """
        params = []
        
        if start_date:
            sql += " AND end_date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND end_date <= %s"
            params.append(end_date)
        
        if sec_ids:
            placeholders = ','.join(['%s'] * len(sec_ids))
            sql += f" AND sec_id IN ({placeholders})"
            params.extend(sec_ids)
        
        sql += " ORDER BY end_date, sec_id"
        
        return self.db_manager.query_dataframe(sql, params)
    
    def get_network_momentum(self, start_date=None, end_date=None, sec_ids=None):
        """
        获取网络动量因子数据
        
        Args:
            start_date (str): 开始日期
            end_date (str): 结束日期
            sec_ids (list): 股票代码列表
            
        Returns:
            pd.DataFrame: 网络动量因子数据
        """
        sql = """
        SELECT sec_id, trade_date, mean_distance, period_mean_distance, factor_rank, factor_group
        FROM network_momentum
        WHERE 1=1
        """
        params = []
        
        if start_date:
            sql += " AND trade_date >= %s"
            params.append(start_date)
        
        if end_date:
            sql += " AND trade_date <= %s"
            params.append(end_date)
        
        if sec_ids:
            placeholders = ','.join(['%s'] * len(sec_ids))
            sql += f" AND sec_id IN ({placeholders})"
            params.extend(sec_ids)
        
        sql += " ORDER BY trade_date, sec_id"
        
        return self.db_manager.query_dataframe(sql, params)
    
    def close(self):
        """关闭数据库连接"""
        self.db_manager.close_connection()

# 示例用法和测试函数
def test_data_manager():
    """测试数据管理器功能"""
    dm = DataManager()
    
    # 测试获取数据
    print("测试获取交易日历数据...")
    calendar_df = dm.get_trade_calendar('2020-01-01', '2020-12-31')
    print(f"获取到 {len(calendar_df)} 条交易日历记录")
    
    print("测试获取股票因子数据...")
    factors_df = dm.get_stock_factors('2020-01-01', '2020-12-31')
    print(f"获取到 {len(factors_df)} 条股票因子记录")
    
    dm.close()

if __name__ == "__main__":
    test_data_manager()
