# -*- coding: utf-8 -*-
"""
测试yuqer数据库表结构
"""

import mysql.connector
import pandas as pd
import json

def test_yuqer_tables():
    """测试yuqer数据库表结构"""
    
    # 加载配置
    with open('para.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    mysql_config = config['mysql_para']
    
    try:
        # 连接到yuqerdata数据库
        connection = mysql.connector.connect(
            host=mysql_config['host'],
            user=mysql_config['user_name'],
            password=mysql_config['pass_wd'],
            port=mysql_config['port'],
            database='yuqerdata',
            auth_plugin='mysql_native_password'
        )
        
        cursor = connection.cursor()
        
        # 检查交易日历表
        print("=== 交易日历表 ===")
        for table_name in ['yuqer_cal', 'yuqer_full']:
            try:
                print(f"\n表: {table_name}")
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                print("列结构:")
                for col in columns:
                    print(f"  {col[0]} - {col[1]}")
                
                # 查看样本数据
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                print("样本数据:")
                for row in sample_data:
                    row_dict = dict(zip(column_names, row))
                    print(f"  {row_dict}")
                    
            except Exception as e:
                print(f"检查表 {table_name} 失败: {e}")
        
        # 检查股票因子表
        print("\n\n=== 股票因子表 ===")
        table_name = 'yq_mktstockfactorsonedayget'
        try:
            print(f"\n表: {table_name}")
            cursor.execute(f"DESCRIBE {table_name}")
            columns = cursor.fetchall()
            print("列结构 (前20列):")
            for i, col in enumerate(columns[:20]):
                print(f"  {col[0]} - {col[1]}")
            
            print(f"总列数: {len(columns)}")
            
            # 查看样本数据
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
            sample_data = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            print("样本数据 (前10列):")
            for row in sample_data:
                row_dict = dict(zip(column_names[:10], row[:10]))
                print(f"  {row_dict}")
                
        except Exception as e:
            print(f"检查表 {table_name} 失败: {e}")
        
        # 检查股票收益率表
        print("\n\n=== 股票收益率表 ===")
        for table_name in ['yq_mktequdadjafget', 'yq_dayprice']:
            try:
                print(f"\n表: {table_name}")
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                print("列结构:")
                for col in columns:
                    print(f"  {col[0]} - {col[1]}")
                
                # 查看样本数据
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                sample_data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                print("样本数据:")
                for row in sample_data:
                    row_dict = dict(zip(column_names, row))
                    print(f"  {row_dict}")
                    
            except Exception as e:
                print(f"检查表 {table_name} 失败: {e}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"连接数据库失败: {e}")

if __name__ == "__main__":
    test_yuqer_tables()
