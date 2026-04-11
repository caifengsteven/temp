# -*- coding: utf-8 -*-
"""
MySQL数据库设置和连接模块
用于网络动量选股策略的数据存储和检索
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import json
import os
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config_file='para.json'):
        """
        初始化数据库管理器
        
        Args:
            config_file (str): 配置文件路径
        """
        self.config_file = config_file
        self.connection = None
        self.load_config()
    
    def load_config(self):
        """从配置文件加载数据库参数"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.mysql_config = config['mysql_para']
                self.mysql_config['host'] = self.mysql_config.get('host', 'localhost')
                self.mysql_config['database'] = self.mysql_config.get('database', 'network_momentum')
                logger.info("数据库配置加载成功")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user_name'],
                password=self.mysql_config['pass_wd'],
                port=self.mysql_config['port'],
                auth_plugin='mysql_native_password'
            )
            
            if self.connection.is_connected():
                logger.info("成功连接到MySQL服务器")
                return True
        except Error as e:
            logger.error(f"连接MySQL失败: {e}")
            return False
    
    def create_database(self):
        """创建数据库（如果不存在）"""
        try:
            cursor = self.connection.cursor()
            database_name = self.mysql_config['database']
            
            # 创建数据库
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {database_name}")
            
            logger.info(f"数据库 {database_name} 创建/选择成功")
            cursor.close()
            return True
        except Error as e:
            logger.error(f"创建数据库失败: {e}")
            return False
    
    def create_tables(self):
        """创建所需的数据表"""
        try:
            cursor = self.connection.cursor()
            
            # 1. 股票基本信息表
            stock_info_table = """
            CREATE TABLE IF NOT EXISTS stock_info (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sec_id VARCHAR(20) NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                stock_name VARCHAR(100),
                list_date DATE,
                delist_date DATE,
                exchange VARCHAR(10),
                industry VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_sec_id (sec_id),
                INDEX idx_ticker (ticker),
                INDEX idx_exchange (exchange)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 2. 交易日历表
            trade_calendar_table = """
            CREATE TABLE IF NOT EXISTS trade_calendar (
                id INT AUTO_INCREMENT PRIMARY KEY,
                calendar_date DATE NOT NULL,
                is_open TINYINT(1) DEFAULT 0,
                is_week_end TINYINT(1) DEFAULT 0,
                is_month_end TINYINT(1) DEFAULT 0,
                exchange_cd VARCHAR(10) DEFAULT 'XSHG',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_date_exchange (calendar_date, exchange_cd),
                INDEX idx_calendar_date (calendar_date),
                INDEX idx_is_open (is_open)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 3. 股票因子数据表
            stock_factors_table = """
            CREATE TABLE IF NOT EXISTS stock_factors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sec_id VARCHAR(20) NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                trade_date DATE NOT NULL,
                mass_index DECIMAL(15,6),
                kdj_j DECIMAL(15,6),
                rsi DECIMAL(15,6),
                cci10 DECIMAL(15,6),
                cmo DECIMAL(15,6),
                mfi DECIMAL(15,6),
                lcap DECIMAL(15,6),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_stock_date (sec_id, trade_date),
                INDEX idx_trade_date (trade_date),
                INDEX idx_ticker (ticker)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 4. 股票收益率数据表
            stock_returns_table = """
            CREATE TABLE IF NOT EXISTS stock_returns (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sec_id VARCHAR(20) NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                end_date DATE NOT NULL,
                chg_pct DECIMAL(15,6),
                close_price DECIMAL(15,4),
                volume BIGINT,
                turnover DECIMAL(20,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_stock_date (sec_id, end_date),
                INDEX idx_end_date (end_date),
                INDEX idx_ticker (ticker)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 5. 网络动量因子表
            network_momentum_table = """
            CREATE TABLE IF NOT EXISTS network_momentum (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sec_id VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                mean_distance DECIMAL(15,6),
                period_mean_distance DECIMAL(15,6),
                factor_rank INT,
                factor_group INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY unique_stock_date (sec_id, trade_date),
                INDEX idx_trade_date (trade_date),
                INDEX idx_factor_rank (factor_rank),
                INDEX idx_factor_group (factor_group)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 6. 回测结果表
            backtest_results_table = """
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                strategy_name VARCHAR(100) NOT NULL,
                trade_date DATE NOT NULL,
                group_num INT,
                cum_ret DECIMAL(15,6),
                period_ret DECIMAL(15,6),
                benchmark_ret DECIMAL(15,6),
                excess_ret DECIMAL(15,6),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_strategy_date (strategy_name, trade_date),
                INDEX idx_trade_date (trade_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # 执行创建表的SQL语句
            tables = [
                ("stock_info", stock_info_table),
                ("trade_calendar", trade_calendar_table),
                ("stock_factors", stock_factors_table),
                ("stock_returns", stock_returns_table),
                ("network_momentum", network_momentum_table),
                ("backtest_results", backtest_results_table)
            ]
            
            for table_name, table_sql in tables:
                cursor.execute(table_sql)
                logger.info(f"表 {table_name} 创建成功")
            
            self.connection.commit()
            cursor.close()
            logger.info("所有数据表创建完成")
            return True
            
        except Error as e:
            logger.error(f"创建数据表失败: {e}")
            return False
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """
        将DataFrame插入到数据库表中
        
        Args:
            df (pd.DataFrame): 要插入的数据
            table_name (str): 目标表名
            if_exists (str): 如果表存在的处理方式 ('append', 'replace', 'fail')
        """
        try:
            # 使用pandas的to_sql方法
            from sqlalchemy import create_engine
            
            # 创建SQLAlchemy引擎
            database_name = self.mysql_config.get('database', 'network_momentum')
            engine_url = f"mysql+mysqlconnector://{self.mysql_config['user_name']}:{self.mysql_config['pass_wd']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{database_name}"
            engine = create_engine(engine_url)
            
            # 插入数据
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi', chunksize=1000)
            logger.info(f"成功插入 {len(df)} 条记录到表 {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"插入数据到表 {table_name} 失败: {e}")
            return False
    
    def query_dataframe(self, sql, params=None):
        """
        执行查询并返回DataFrame
        
        Args:
            sql (str): SQL查询语句
            params (tuple): 查询参数
            
        Returns:
            pd.DataFrame: 查询结果
        """
        try:
            df = pd.read_sql(sql, self.connection, params=params)
            return df
        except Exception as e:
            logger.error(f"查询数据失败: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")

def setup_database():
    """设置数据库的主函数"""
    db_manager = DatabaseManager()
    
    # 连接数据库
    if not db_manager.connect():
        return False
    
    # 创建数据库
    if not db_manager.create_database():
        return False
    
    # 重新连接到指定数据库
    db_manager.close_connection()

    try:
        db_manager.connection = mysql.connector.connect(
            host=db_manager.mysql_config['host'],
            user=db_manager.mysql_config['user_name'],
            password=db_manager.mysql_config['pass_wd'],
            port=db_manager.mysql_config['port'],
            database='network_momentum',
            auth_plugin='mysql_native_password'
        )
        logger.info("成功连接到network_momentum数据库")
    except Error as e:
        logger.error(f"连接到数据库失败: {e}")
        return False
    
    # 创建数据表
    if not db_manager.create_tables():
        return False
    
    logger.info("数据库设置完成！")
    return db_manager

if __name__ == "__main__":
    # 运行数据库设置
    db_manager = setup_database()
    if db_manager:
        print("数据库设置成功！")
        db_manager.close_connection()
    else:
        print("数据库设置失败！")
