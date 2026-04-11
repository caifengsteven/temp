# -*- coding: utf-8 -*-
"""
网络动量策略数据更新需求分析
分析当前数据状况和更新需求
"""

from datayes_mysql_api import DataAPI
import pandas as pd
import numpy as np

def analyze_data_requirements():
    """分析数据更新需求"""
    
    print("=" * 80)
    print("网络动量策略数据更新需求分析")
    print("=" * 80)
    
    # 1. 当前网络动量数据状况
    print("\n1. 当前网络动量数据状况")
    print("-" * 50)
    
    try:
        network_momentum_df = pd.read_csv('网络动量_长期分析结果.csv')
        print(f"✓ 现有网络动量数据: {len(network_momentum_df)} 条记录")
        print(f"  股票数量: {network_momentum_df['secID'].nunique()}")
        print(f"  日期范围: {network_momentum_df['tradeDate'].min()} 到 {network_momentum_df['tradeDate'].max()}")
        print(f"  数据结构: {list(network_momentum_df.columns)}")
        
        # 分析数据密度
        date_counts = network_momentum_df['tradeDate'].value_counts().sort_index()
        print(f"  平均每日股票数: {date_counts.mean():.0f}")
        print(f"  数据覆盖期数: {len(date_counts)} 个交易周")
        
    except Exception as e:
        print(f"✗ 无法读取现有网络动量数据: {e}")
        return
    
    # 2. 数据库可用数据范围
    print("\n2. 数据库可用数据范围")
    print("-" * 50)
    
    # 检查因子数据范围
    print("检查股票因子数据...")
    factor_dates = []
    test_years = ['2020', '2021', '2022', '2023', '2024']
    
    for year in test_years:
        test_date = f"{year}-01-05"
        try:
            factor_data = DataAPI.MktStockFactorsOneDayGet(
                tradeDate=test_date,
                field=['secID', 'ticker', 'tradeDate', 'MassIndex', 'KDJ_J', 'RSI']
            )
            if len(factor_data) > 0:
                factor_dates.append((test_date, len(factor_data)))
                print(f"  ✓ {test_date}: {len(factor_data)} 条因子记录")
            else:
                print(f"  ✗ {test_date}: 无数据")
        except Exception as e:
            print(f"  ✗ {test_date}: 查询失败 - {e}")
    
    # 检查收益率数据范围
    print("\n检查股票收益率数据...")
    return_periods = []
    for year in test_years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-01-31"
        try:
            return_data = DataAPI.MktEquwAdjGet(
                beginDate=start_date, 
                endDate=end_date, 
                field="secID,endDate,chgPct"
            )
            if len(return_data) > 0:
                return_periods.append((year, len(return_data), return_data['secID'].nunique()))
                print(f"  ✓ {year}年1月: {len(return_data)} 条记录, {return_data['secID'].nunique()} 只股票")
            else:
                print(f"  ✗ {year}年1月: 无数据")
        except Exception as e:
            print(f"  ✗ {year}年1月: 查询失败 - {e}")
    
    # 3. 数据更新需求分析
    print("\n3. 数据更新需求分析")
    print("-" * 50)
    
    current_end_date = network_momentum_df['tradeDate'].max()
    print(f"当前网络动量数据截止: {current_end_date}")
    
    if factor_dates:
        latest_factor_date = max(factor_dates, key=lambda x: x[0])[0]
        print(f"数据库最新因子数据: {latest_factor_date}")
        
        # 计算需要更新的时间范围
        from datetime import datetime, timedelta
        current_date = datetime.strptime(current_end_date, '%Y-%m-%d')
        latest_date = datetime.strptime(latest_factor_date, '%Y-%m-%d')
        
        if latest_date > current_date:
            days_to_update = (latest_date - current_date).days
            print(f"需要更新的时间跨度: {days_to_update} 天")
            
            # 估算需要处理的数据量
            if factor_dates:
                avg_stocks_per_day = np.mean([count for _, count in factor_dates])
                estimated_records = avg_stocks_per_day * (days_to_update // 7)  # 按周计算
                print(f"预估需要处理的新记录数: {estimated_records:.0f} 条")
        else:
            print("✓ 当前数据已是最新")
    
    # 4. 具体更新步骤
    print("\n4. 数据更新具体步骤")
    print("-" * 50)
    
    print("步骤1: 获取新的交易日历数据")
    print("  - 获取2019-09-01到2024-12-31的周末交易日")
    print("  - 预估约260个新的周末日期")
    
    print("\n步骤2: 获取新的股票因子数据")
    print("  - 需要的因子: MassIndex, KDJ_J, RSI, CCI10, CMO, MFI")
    print("  - 时间范围: 2019-09-01 到 2024-01-05")
    print("  - 预估数据量: 约200万条记录")
    
    print("\n步骤3: 因子标准化处理")
    print("  - 按股票分组进行52周滚动标准化")
    print("  - 处理缺失值和异常值")
    
    print("\n步骤4: 计算网络距离矩阵")
    print("  - 每个交易日计算股票间的欧几里得距离")
    print("  - 计算每只股票与其他股票的平均距离")
    
    print("\n步骤5: 计算期间平均网络动量")
    print("  - 计算过去4周的滚动平均距离")
    print("  - 生成最终的网络动量因子")
    
    print("\n步骤6: 获取对应的收益率数据")
    print("  - 获取2019-09-01到2023-12-31的股票收益率")
    print("  - 计算下期收益率用于回测")
    
    # 5. 资源需求估算
    print("\n5. 资源需求估算")
    print("-" * 50)
    
    print("计算资源需求:")
    print("  - 内存需求: 约8-16GB (处理大型距离矩阵)")
    print("  - 存储需求: 约2-3GB (新的网络动量数据)")
    print("  - 处理时间: 约3-5小时 (取决于硬件性能)")
    
    print("\n数据库查询需求:")
    print("  - 因子数据查询: 约260次 (每个周末一次)")
    print("  - 收益率数据查询: 约5次 (按年分批)")
    print("  - 网络连接稳定性要求较高")
    
    # 6. 更新后的预期效果
    print("\n6. 更新后的预期效果")
    print("-" * 50)
    
    print("数据覆盖范围:")
    print("  - 时间跨度: 2015-2024 (约9年)")
    print("  - 回测期间: 2015-2023 (约8年)")
    print("  - 样本外测试: 2024年数据")
    
    print("\n策略性能预期:")
    print("  - 更长的回测期间提供更可靠的统计结果")
    print("  - 包含更多市场周期(牛市、熊市、震荡市)")
    print("  - 可以验证策略在不同市场环境下的稳定性")
    
    print("\n" + "=" * 80)
    print("总结建议")
    print("=" * 80)
    
    if factor_dates and return_periods:
        print("✓ 数据库包含足够的新数据，建议进行更新")
        print("✓ 更新后可将回测期间扩展到2023年")
        print("✓ 预期年化收益率可能进一步提升")
        print("\n推荐更新方案:")
        print("1. 分批处理数据以避免内存溢出")
        print("2. 保存中间结果以便断点续传")
        print("3. 并行处理距离计算以提高效率")
        print("4. 定期备份数据以防意外丢失")
    else:
        print("✗ 数据库数据不足，暂不建议更新")
        print("建议检查数据源或联系数据提供商")

if __name__ == "__main__":
    analyze_data_requirements()
