# -*- coding: utf-8 -*-
"""
检查yuqer数据库中可用的因子
"""

import mysql.connector
import json

def check_available_factors():
    """检查可用的因子"""
    
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
        
        # 获取yq_mktstockfactorsonedayget表的所有列
        cursor.execute("DESCRIBE yq_mktstockfactorsonedayget")
        columns = cursor.fetchall()
        
        # 我们需要的因子
        needed_factors = ['MassIndex', 'KDJ_J', 'RSI', 'CCI10', 'CMO', 'MFI', 'LCAP']
        
        print("需要的因子:")
        for factor in needed_factors:
            print(f"  - {factor}")
        
        print(f"\n表中所有列 (共{len(columns)}列):")
        available_factors = []
        for col in columns:
            col_name = col[0]
            print(f"  {col_name}")
            if col_name in needed_factors:
                available_factors.append(col_name)
        
        print(f"\n找到的匹配因子:")
        for factor in available_factors:
            print(f"  ✓ {factor}")
        
        print(f"\n缺失的因子:")
        missing_factors = [f for f in needed_factors if f not in available_factors]
        for factor in missing_factors:
            print(f"  ✗ {factor}")
        
        # 查找可能的替代因子
        print(f"\n可能的替代因子:")
        factor_keywords = {
            'MassIndex': ['mass', 'index'],
            'KDJ_J': ['kdj', 'j'],
            'RSI': ['rsi'],
            'CCI10': ['cci'],
            'CMO': ['cmo'],
            'MFI': ['mfi'],
            'LCAP': ['lcap', 'cap', 'market']
        }
        
        for needed_factor, keywords in factor_keywords.items():
            if needed_factor not in available_factors:
                candidates = []
                for col in columns:
                    col_name = col[0].lower()
                    if any(keyword.lower() in col_name for keyword in keywords):
                        candidates.append(col[0])
                
                if candidates:
                    print(f"  {needed_factor} 的可能替代: {candidates}")
        
        # 测试获取一些数据
        if available_factors:
            print(f"\n测试获取数据...")
            test_fields = ['secID', 'ticker', 'tradeDate'] + available_factors[:3]
            field_str = ', '.join(test_fields)
            
            cursor.execute(f"""
                SELECT {field_str} 
                FROM yq_mktstockfactorsonedayget 
                WHERE tradeDate = '2019-01-04' 
                LIMIT 5
            """)
            
            sample_data = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            print("样本数据:")
            for row in sample_data:
                row_dict = dict(zip(column_names, row))
                print(f"  {row_dict}")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"检查因子失败: {e}")

if __name__ == "__main__":
    check_available_factors()
