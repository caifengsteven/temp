# -*- coding: utf-8 -*-
"""
测试MySQL连接
"""

import json
import sys

def test_mysql_connection():
    """测试MySQL连接"""
    try:
        # 加载配置
        with open('para.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        mysql_config = config['mysql_para']
        print(f"尝试连接到MySQL: {mysql_config['host']}:{mysql_config['port']}")
        print(f"用户名: {mysql_config['user_name']}")
        
        # 尝试不同的连接方式
        
        # 方法1: mysql-connector-python
        try:
            import mysql.connector
            print("\n方法1: 使用mysql-connector-python")
            
            connection = mysql.connector.connect(
                host=mysql_config['host'],
                user=mysql_config['user_name'],
                password=mysql_config['pass_wd'],
                port=mysql_config['port'],
                auth_plugin='mysql_native_password'
            )
            
            if connection.is_connected():
                print("✓ mysql-connector-python 连接成功!")
                cursor = connection.cursor()
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()
                print(f"MySQL版本: {version[0]}")
                cursor.close()
                connection.close()
                return True
            
        except Exception as e:
            print(f"✗ mysql-connector-python 连接失败: {e}")
        
        # 方法2: PyMySQL
        try:
            import pymysql
            print("\n方法2: 使用PyMySQL")
            
            connection = pymysql.connect(
                host=mysql_config['host'],
                user=mysql_config['user_name'],
                password=mysql_config['pass_wd'],
                port=mysql_config['port'],
                charset='utf8mb4'
            )
            
            print("✓ PyMySQL 连接成功!")
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"MySQL版本: {version[0]}")
            cursor.close()
            connection.close()
            return True
            
        except Exception as e:
            print(f"✗ PyMySQL 连接失败: {e}")
        
        return False
        
    except Exception as e:
        print(f"配置加载失败: {e}")
        return False

def install_mysql_packages():
    """安装MySQL相关包"""
    import subprocess
    
    packages = [
        'mysql-connector-python',
        'pymysql',
        'sqlalchemy'
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} 安装失败: {e}")

if __name__ == "__main__":
    print("MySQL连接测试")
    print("=" * 40)
    
    # 首先尝试安装必要的包
    print("1. 安装MySQL连接包...")
    install_mysql_packages()
    
    print("\n2. 测试MySQL连接...")
    success = test_mysql_connection()
    
    if success:
        print("\n✓ MySQL连接测试成功!")
    else:
        print("\n✗ MySQL连接测试失败!")
        print("\n可能的解决方案:")
        print("1. 检查MySQL服务是否正在运行")
        print("2. 检查用户名和密码是否正确")
        print("3. 检查MySQL是否允许本地连接")
        print("4. 尝试重启MySQL服务")
