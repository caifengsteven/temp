# -*- coding: utf-8 -*-
"""
DataYes MySQL API 替代模块
用于从MySQL数据库中的yuqer数据替代原始的DataYes API调用
"""

import mysql.connector
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

class DataAPI:
    """DataYes API的MySQL替代实现"""
    
    def __init__(self, config_file='para.json'):
        self.config_file = config_file
        self.connection = None
        self.load_config()
        self.connect()
    
    def load_config(self):
        """加载数据库配置"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.mysql_config = config['mysql_para']
    
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user_name'],
                password=self.mysql_config['pass_wd'],
                port=self.mysql_config['port'],
                database='yuqerdata',  # 使用yuqerdata数据库
                auth_plugin='mysql_native_password'
            )
            logger.info("成功连接到yuqer数据库")
        except Exception as e:
            logger.error(f"连接yuqer数据库失败: {e}")
            # 尝试连接到其他可能的数据库名
            try:
                self.connection = mysql.connector.connect(
                    host=self.mysql_config['host'],
                    user=self.mysql_config['user_name'],
                    password=self.mysql_config['pass_wd'],
                    port=self.mysql_config['port'],
                    auth_plugin='mysql_native_password'
                )
                logger.info("连接到MySQL服务器，将查找yuqer相关数据库")
            except Exception as e2:
                logger.error(f"连接MySQL失败: {e2}")
                raise
    
    def find_yuqer_database(self):
        """查找yuqer相关的数据库"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW DATABASES")
            databases = cursor.fetchall()
            
            yuqer_dbs = []
            for db in databases:
                db_name = db[0].lower()
                if 'yuqer' in db_name or 'datayes' in db_name or 'uqer' in db_name:
                    yuqer_dbs.append(db[0])
            
            cursor.close()
            
            if yuqer_dbs:
                logger.info(f"找到yuqer相关数据库: {yuqer_dbs}")
                return yuqer_dbs[0]  # 使用第一个找到的
            else:
                logger.warning("未找到yuqer相关数据库")
                return None
                
        except Exception as e:
            logger.error(f"查找数据库失败: {e}")
            return None
    
    def execute_query(self, sql, params=None):
        """执行SQL查询"""
        try:
            df = pd.read_sql(sql, self.connection, params=params)
            return df
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def TradeCalGet(exchangeCD="XSHG", beginDate=None, endDate=None, field="", pandas="1"):
        """
        获取交易日历
        替代 DataAPI.TradeCalGet
        """
        api = DataAPI()
        
        # 使用yuqerdata数据库中的交易日历表
        possible_tables = [
            'yuqer_cal', 'yuqer_full', 'uq_yuqer_cal'
        ]
        
        for table_name in possible_tables:
            try:
                sql = f"""
                SELECT calendarDate, isOpen, isWeekEnd, isMonthEnd, exchangeCD
                FROM {table_name}
                WHERE exchangeCD = %s
                """
                params = [exchangeCD]
                
                if beginDate:
                    sql += " AND calendarDate >= %s"
                    params.append(beginDate)
                
                if endDate:
                    sql += " AND calendarDate <= %s"
                    params.append(endDate)
                
                sql += " ORDER BY calendarDate"
                
                df = api.execute_query(sql, params)
                if len(df) > 0:
                    logger.info(f"从表 {table_name} 获取到 {len(df)} 条交易日历数据")
                    api.close()
                    return df
                    
            except Exception as e:
                logger.debug(f"表 {table_name} 查询失败: {e}")
                continue
        
        logger.warning("未找到交易日历数据")
        api.close()
        return pd.DataFrame()
    
    @staticmethod
    def MktStockFactorsOneDayGet(tradeDate=None, secID="", ticker="", field=None, pandas="1"):
        """
        获取股票因子数据
        替代 DataAPI.MktStockFactorsOneDayGet
        """
        api = DataAPI()
        
        # 使用yuqerdata数据库中的股票因子表
        possible_tables = [
            'yq_mktstockfactorsonedayget', 'yq_mktstockfactorsonedayproget'
        ]
        
        if field is None:
            field = ['secID', 'ticker', 'tradeDate', 'MassIndex', 'KDJ_J', 'RSI', 'CCI10', 'CMO', 'MFI', 'LCAP']
        
        field_str = ', '.join(field) if isinstance(field, list) else field
        
        for table_name in possible_tables:
            try:
                sql = f"SELECT {field_str} FROM {table_name} WHERE 1=1"
                params = []
                
                if tradeDate:
                    sql += " AND tradeDate = %s"
                    params.append(tradeDate)
                
                if secID:
                    sql += " AND secID = %s"
                    params.append(secID)
                
                if ticker:
                    sql += " AND ticker = %s"
                    params.append(ticker)
                
                df = api.execute_query(sql, params)
                if len(df) > 0:
                    logger.info(f"从表 {table_name} 获取到 {len(df)} 条因子数据")
                    api.close()
                    return df
                    
            except Exception as e:
                logger.debug(f"表 {table_name} 查询失败: {e}")
                continue
        
        logger.warning("未找到股票因子数据")
        api.close()
        return pd.DataFrame()
    
    @staticmethod
    def MktEquwAdjGet(beginDate=None, endDate=None, secID="", field=None, pandas="1"):
        """
        获取股票收益率数据
        替代 DataAPI.MktEquwAdjGet
        """
        api = DataAPI()
        
        # 使用yuqerdata数据库中的股票收益率表
        possible_tables = [
            'yq_mktequdadjafget', 'yq_dayprice'
        ]

        for table_name in possible_tables:
            try:
                if table_name == 'yq_mktequdadjafget':
                    # 对于yq_mktequdadjafget表，需要计算chgPct
                    if field is None or "chgPct" in field:
                        sql = f"""
                        SELECT secID, tradeDate as endDate,
                               (closePrice - preClosePrice) / preClosePrice as chgPct,
                               closePrice, turnoverVol as volume, turnoverValue as turnover
                        FROM {table_name} WHERE 1=1
                        """
                    else:
                        field_str = field if isinstance(field, str) else ', '.join(field)
                        field_str = field_str.replace('endDate', 'tradeDate as endDate')
                        field_str = field_str.replace('volume', 'turnoverVol as volume')
                        field_str = field_str.replace('turnover', 'turnoverValue as turnover')
                        sql = f"SELECT {field_str} FROM {table_name} WHERE 1=1"
                elif table_name == 'yq_dayprice':
                    # 对于yq_dayprice表，已经有chgPct字段
                    if field is None:
                        sql = f"""
                        SELECT CONCAT(symbol, '.XSHE') as secID, tradeDate as endDate, chgPct,
                               closePrice, turnoverVol as volume, turnoverValue as turnover
                        FROM {table_name} WHERE 1=1
                        """
                    else:
                        field_str = field if isinstance(field, str) else ', '.join(field)
                        field_str = field_str.replace('secID', 'CONCAT(symbol, ".XSHE") as secID')
                        field_str = field_str.replace('endDate', 'tradeDate as endDate')
                        field_str = field_str.replace('volume', 'turnoverVol as volume')
                        field_str = field_str.replace('turnover', 'turnoverValue as turnover')
                        sql = f"SELECT {field_str} FROM {table_name} WHERE 1=1"

                params = []

                if beginDate:
                    sql += " AND tradeDate >= %s"
                    params.append(beginDate)

                if endDate:
                    sql += " AND tradeDate <= %s"
                    params.append(endDate)

                if secID and table_name == 'yq_mktequdadjafget':
                    sql += " AND secID = %s"
                    params.append(secID)
                elif secID and table_name == 'yq_dayprice':
                    # 从secID中提取symbol
                    symbol = secID.split('.')[0] if '.' in secID else secID
                    sql += " AND symbol = %s"
                    params.append(symbol)

                sql += " ORDER BY tradeDate, secID"
                
                df = api.execute_query(sql, params)
                if len(df) > 0:
                    logger.info(f"从表 {table_name} 获取到 {len(df)} 条收益率数据")
                    api.close()
                    return df
                    
            except Exception as e:
                logger.debug(f"表 {table_name} 查询失败: {e}")
                continue
        
        logger.warning("未找到股票收益率数据")
        api.close()
        return pd.DataFrame()
    
    def close(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")

def check_yuqer_tables():
    """检查yuqer数据库中的表"""
    try:
        api = DataAPI()
        
        # 如果直接连接yuqer失败，查找相关数据库
        if not api.connection.is_connected():
            yuqer_db = api.find_yuqer_database()
            if yuqer_db:
                api.connection.database = yuqer_db
        
        cursor = api.connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        print("yuqer数据库中的表:")
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
            
            # 检查表结构
            try:
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                print(f"    列数: {len(columns)}")
                
                # 检查数据量
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"    记录数: {count}")
                
            except Exception as e:
                print(f"    错误: {e}")
        
        cursor.close()
        api.close()
        
    except Exception as e:
        print(f"检查yuqer数据库失败: {e}")

if __name__ == "__main__":
    check_yuqer_tables()
