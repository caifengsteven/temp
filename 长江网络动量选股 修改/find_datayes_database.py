# -*- coding: utf-8 -*-
"""
查找DataYes相关的数据库
"""

import mysql.connector
import json

def find_datayes_database():
    """查找DataYes相关的数据库"""
    
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
        print("所有可用的数据库:")
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        
        datayes_related = []
        for db in databases:
            db_name = db[0]
            print(f"  - {db_name}")
            
            # 检查是否包含DataYes相关关键词
            if any(keyword in db_name.lower() for keyword in ['yuqer', 'datayes', 'uqer', 'quant', 'stock', 'market']):
                datayes_related.append(db_name)
        
        print(f"\n可能的DataYes相关数据库: {datayes_related}")
        
        # 检查每个数据库中的表
        for db_name in databases:
            db_name = db_name[0]
            if db_name in ['information_schema', 'performance_schema', 'mysql', 'sys']:
                continue
                
            print(f"\n检查数据库: {db_name}")
            try:
                cursor.execute(f"USE {db_name}")
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
                if tables:
                    print(f"  表数量: {len(tables)}")
                    
                    # 查找可能的DataYes表
                    datayes_tables = []
                    for table in tables:
                        table_name = table[0]
                        if any(keyword in table_name.lower() for keyword in 
                               ['mkt', 'stock', 'trade', 'factor', 'equ', 'cal', 'price']):
                            datayes_tables.append(table_name)
                    
                    if datayes_tables:
                        print(f"  可能的DataYes表: {datayes_tables[:10]}")  # 只显示前10个
                        
                        # 检查第一个表的结构
                        if datayes_tables:
                            sample_table = datayes_tables[0]
                            print(f"  检查表 {sample_table} 的结构:")
                            try:
                                cursor.execute(f"DESCRIBE {sample_table}")
                                columns = cursor.fetchall()
                                column_names = [col[0] for col in columns]
                                print(f"    列: {column_names}")
                                
                                # 检查数据量
                                cursor.execute(f"SELECT COUNT(*) FROM {sample_table}")
                                count = cursor.fetchone()[0]
                                print(f"    记录数: {count}")
                                
                                if count > 0:
                                    print(f"    样本数据:")
                                    cursor.execute(f"SELECT * FROM {sample_table} LIMIT 2")
                                    sample_data = cursor.fetchall()
                                    for row in sample_data:
                                        print(f"      {row}")
                                
                            except Exception as e:
                                print(f"    检查表结构失败: {e}")
                
            except Exception as e:
                print(f"  检查数据库 {db_name} 失败: {e}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"连接数据库失败: {e}")

if __name__ == "__main__":
    find_datayes_database()
