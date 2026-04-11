# -*- coding: utf-8 -*-
"""
MySQL数据库设置脚本
用于初始化网络动量选股策略的数据库环境
"""

import os
import sys
import json
import subprocess
import logging
from database_setup import setup_database
from data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_mysql_service():
    """检查MySQL服务是否运行"""
    try:
        import mysql.connector
        
        # 读取配置
        with open('para.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        mysql_config = config['mysql_para']
        
        # 尝试连接
        connection = mysql.connector.connect(
            host=mysql_config.get('host', 'localhost'),
            user=mysql_config['user_name'],
            password=mysql_config['pass_wd'],
            port=mysql_config['port']
        )
        
        if connection.is_connected():
            connection.close()
            logger.info("✓ MySQL服务运行正常")
            return True
        else:
            logger.error("✗ 无法连接到MySQL服务")
            return False
            
    except Exception as e:
        logger.error(f"✗ MySQL连接失败: {e}")
        return False

def install_requirements():
    """安装Python依赖包"""
    try:
        logger.info("正在安装Python依赖包...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✓ Python依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ 依赖包安装失败: {e}")
        return False

def setup_mysql_database():
    """设置MySQL数据库"""
    try:
        logger.info("正在设置MySQL数据库...")
        db_manager = setup_database()
        if db_manager:
            db_manager.close_connection()
            logger.info("✓ MySQL数据库设置完成")
            return True
        else:
            logger.error("✗ MySQL数据库设置失败")
            return False
    except Exception as e:
        logger.error(f"✗ 数据库设置出错: {e}")
        return False

def test_database_connection():
    """测试数据库连接和基本功能"""
    try:
        logger.info("正在测试数据库连接...")
        dm = DataManager()
        
        # 测试获取交易日历
        calendar_df = dm.get_trade_calendar('2020-01-01', '2020-01-31')
        logger.info(f"✓ 数据库连接测试成功，获取到 {len(calendar_df)} 条交易日历记录")
        
        dm.close()
        return True
    except Exception as e:
        logger.error(f"✗ 数据库连接测试失败: {e}")
        return False

def create_lib_directory():
    """创建lib目录（如果需要）"""
    lib_dir = "lib"
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)
        logger.info(f"✓ 创建了 {lib_dir} 目录")
        
        # 创建一个简单的quant_util模块
        quant_util_content = '''# -*- coding: utf-8 -*-
"""
量化工具模块
简化版本，用于网络动量策略
"""

import pandas as pd
import numpy as np

def simple_group_backtest(factor_df, return_df, factor_col, return_col, ngrp=5):
    """简单的分组回测"""
    # 这里是一个简化的实现
    # 在实际使用中，您需要实现完整的分组回测逻辑
    result_df = pd.DataFrame({
        'tradeDate': factor_df['tradeDate'].unique(),
        'group': 0,
        'cum_ret': 1.0
    })
    return result_df

def long_short_backtest(factor_df, return_df, factor_col, return_col, direction=1, commission=0.0):
    """多空回测"""
    # 这里是一个简化的实现
    # 在实际使用中，您需要实现完整的多空回测逻辑
    result_df = pd.DataFrame({
        'tradeDate': factor_df['tradeDate'].unique(),
        'cum_ret': 1.0,
        'period_ret': 0.0
    })
    bt_df = pd.DataFrame()
    return result_df, bt_df

def signal_grouping(factor_df, factor_col, ngrp=5):
    """信号分组"""
    factor_df_copy = factor_df.copy()
    factor_df_copy['group'] = 0  # 简化实现
    return factor_df_copy

def netralize_dframe(factor_df, factor_cols, exclude_style=None):
    """因子中性化"""
    # 简化实现，直接返回原数据
    return factor_df
'''
        
        with open(os.path.join(lib_dir, 'quant_util.py'), 'w', encoding='utf-8') as f:
            f.write(quant_util_content)
        
        # 创建__init__.py文件
        with open(os.path.join(lib_dir, '__init__.py'), 'w', encoding='utf-8') as f:
            f.write('# Quantitative utilities library\n')
        
        logger.info("✓ 创建了简化的quant_util模块")
    else:
        logger.info("✓ lib目录已存在")

def main():
    """主设置函数"""
    print("=" * 60)
    print("网络动量选股策略 - MySQL数据库设置")
    print("=" * 60)
    
    # 1. 检查配置文件
    if not os.path.exists('para.json'):
        logger.error("✗ 配置文件 para.json 不存在")
        return False
    
    logger.info("✓ 配置文件存在")
    
    # 2. 安装依赖包
    if not install_requirements():
        return False
    
    # 3. 检查MySQL服务
    if not check_mysql_service():
        logger.error("请确保MySQL服务正在运行，并检查para.json中的连接参数")
        return False
    
    # 4. 设置数据库
    if not setup_mysql_database():
        return False
    
    # 5. 创建lib目录和工具模块
    create_lib_directory()
    
    # 6. 测试数据库连接
    if not test_database_connection():
        return False
    
    print("\n" + "=" * 60)
    print("✓ MySQL数据库设置完成！")
    print("=" * 60)
    print("\n接下来您可以:")
    print("1. 运行 python 网络动量_mysql.py 来执行策略")
    print("2. 使用 DataManager 类来管理您的数据")
    print("3. 导入真实的股票数据到数据库中")
    print("\n注意: 当前使用的是示例数据，请替换为真实的股票数据")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n设置过程中遇到错误，请检查上述日志信息")
        sys.exit(1)
