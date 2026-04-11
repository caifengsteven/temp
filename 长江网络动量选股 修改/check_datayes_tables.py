# -*- coding: utf-8 -*-
"""
检查MySQL数据库中的DataYes表结构
"""

import mysql.connector
import pandas as pd
import json

def check_datayes_tables():
    """检查DataYes数据库表"""
    
    # 加载配置
    with open('para.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    mysql_config = config['mysql_para']
    
    try:
        # 连接到MySQL
        connection = mysql.connector.connect(
            host=mysql_config['host'],
            user=mysql_config['user_name'],
            password=mysql_config['pass_wd'],
            port=mysql_config['port'],
            auth_plugin='mysql_native_password'
        )
        
        cursor = connection.cursor()
        
        # 显示所有数据库
        print("可用的数据库:")
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        for db in databases:
            print(f"  - {db[0]}")
        
        print("\n" + "="*50)
        
        # 检查每个数据库中的表
        for db in databases:
            db_name = db[0]
            if db_name in ['information_schema', 'performance_schema', 'mysql', 'sys']:
                continue
                
            print(f"\n数据库: {db_name}")
            cursor.execute(f"USE {db_name}")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            if tables:
                print(f"  表数量: {len(tables)}")
                for table in tables[:10]:  # 只显示前10个表
                    table_name = table[0]
                    print(f"    - {table_name}")
                    
                    # 检查表结构
                    try:
                        cursor.execute(f"DESCRIBE {table_name}")
                        columns = cursor.fetchall()
                        print(f"      列数: {len(columns)}")
                        
                        # 检查数据量
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"      记录数: {count}")
                        
                        # 如果表名包含常见的金融数据关键词，显示样本数据
                        if any(keyword in table_name.lower() for keyword in ['mkt', 'stock', 'trade', 'factor', 'price']):
                            print(f"      样本数据:")
                            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                            sample_data = cursor.fetchall()
                            column_names = [desc[0] for desc in cursor.description]
                            
                            for row in sample_data:
                                row_dict = dict(zip(column_names, row))
                                print(f"        {row_dict}")
                        
                    except Exception as e:
                        print(f"      错误: {e}")
                    
                    print()
                
                if len(tables) > 10:
                    print(f"    ... 还有 {len(tables) - 10} 个表")
            else:
                print("  没有表")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"连接数据库失败: {e}")

if __name__ == "__main__":
    check_datayes_tables()
