%S28 所有计算步骤逐个自动计算

%创建空数据库表
run_create_table()

run_pause()
%验证  股指期货日内收益分布特征
run_M_check_000985_update()
sprintf('完成：%s','验证  股指期货日内收益分布特征')
run_pause()

%验证 300、IF日内收益分布，基差、日持仓量变化(2016年1月开始)
run_M_static_300()
sprintf('完成：%s','验证 300、IF日内收益分布，基差、日持仓量变化(2016年1月开始)')
run_pause()

%验证 日持仓量变化2016年12月开始  和文献结果相同
run_M_static_300_volume();
sprintf('完成：%s','验证 日持仓量变化2016年12月开始  和文献结果相同')
run_pause()

%验证 使用wind数据验证IF、IH、IC的日内收益，start时间为2016年12月
run_M_static();
sprintf('完成：%s','验证 使用wind数据验证IF、IH、IC的日内收益，start时间为2016年12月')
run_pause()

%验证 不同时段累计收益率
run_M_rule1_update()
sprintf('完成：%s','验证 不同时段累计收益率')
run_pause()

%合成并验证因子2.1 收盘折溢价
run_M_21()
sprintf('完成：%s','合成并验证因子2.1 收盘折溢价')
run_pause()

%合成并验证因子2.2 收盘折溢价
run_M_22();
sprintf('完成：%s','合成并验证因子2.2 收盘折溢价')
run_pause()

%合成并验证因子2.3 收盘折溢价
run_M_23();
sprintf('完成：%s','合成并验证因子2.3 收盘折溢价')
run_pause()

%将2.3的15分钟、30分钟因子回测图放在同一个图上
run_M_23_figure();
sprintf('完成：%s','将2.3的15分钟、30分钟因子回测图放在同一个图上')
run_pause()

%合成并验证因子2.4 收盘折溢价
run_M_24();
sprintf('完成：%s','合成并验证因子2.4 收盘折溢价')
run_pause()

%验证3.1 复合策略 1——双因子信号取并集
run_M_com_2factor();
sprintf('完成：%s','验证3.1 复合策略 1——双因子信号取并集')
run_pause()

%验证3.2 复合策略 2——双因子信号取交集
run_M_com_2factor_intersect();
sprintf('完成：%s','验证3.2 复合策略 2——双因子信号取交集')
run_pause()

%验证3.3 复合策略 3——三因子策略
run_M_com_3factor();
sprintf('完成：%s','验证3.3 复合策略 3——三因子策略')
run_pause()

%验证 4.1 策略对交易成本的敏感性-复合策略1
run_M_com_2factor_feeTest()
sprintf('完成：%s','验证 4.1 策略对交易成本的敏感性-复合策略1')
run_pause()

%验证 4.1 策略对交易成本的敏感性-复合策略2
run_M_com_2factor_intersect_feeTest();
sprintf('完成：%s','验证 4.1 策略对交易成本的敏感性-复合策略2')
run_pause()

%验证 4.1 策略对交易成本的敏感性-复合策略3
run_M_com_3factor_feeTest()
sprintf('完成：%s','验证 4.1 策略对交易成本的敏感性-复合策略3')
run_pause()

%验证 4.2 策略对对平仓时点的敏感性-复合策略1
run_M_com_2factor_closetimeTest();
sprintf('完成：%s','验证 4.2 策略对对平仓时点的敏感性-复合策略1')
run_pause()

%验证 4.2 策略对对平仓时点的敏感性-复合策略2
run_M_com_2factor_intersect_closetimeTest();
sprintf('完成：%s','验证 4.2 策略对对平仓时点的敏感性-复合策略2')
run_pause()

%验证 4.2 策略对对平仓时点的敏感性-复合策略3
run_M_com_3factor_closetimeTest()
sprintf('完成：%s','验证 4.2 策略对对平仓时点的敏感性-复合策略3')
run_pause()

%验证 4.3 精细化回测-数据准备
run_M_factor_price();
sprintf('完成：%s','验证 4.3 精细化回测-数据准备')
run_pause()

%验证 4.3 精细化回测-复合策略1
run_M_com_2factor_final();
sprintf('完成：%s','验证 4.3 精细化回测-复合策略1')
run_pause()

%验证 4.3 精细化回测-复合策略2
run_M_com_2factor_intersect_final();
sprintf('完成：%s','验证 4.3 精细化回测-复合策略2')
run_pause()

%验证 4.3 精细化回测-复合策略3
run_M_com_3factor_final();
sprintf('完成：%s','验证 4.3 精细化回测-复合策略3')
run_pause()

%精细化回测结果比较图  含参数统计表
run_M_figure_final();
sprintf('完成：%s','精细化回测结果比较图  含参数统计表')
run_pause()

function run_create_table()
    M_S28_create_table
end

function run_M_check_000985_update()
    M_check_000985_update
end
function run_M_static_300()
    M_static_300
end
function run_M_static_300_volume()
    M_static_300_volume;
end
function run_M_static()
    M_static
end
function run_M_rule1_update()
    M_rule1_update;
end
function run_M_21()
    %合成因子并保存
    M_factor_21;
    %验证因子
    M_rule21_update;
end
function run_M_22()
    %合成因子并保存
    M_factor_22;
    %验证因子
    M_rule22_update;
end
function run_M_23()
    %合成因子并保存
    M_factor_23;
    %验证因子
    M_rule23_update;
end
function run_M_23_figure()
    M_rule23_update_figure
end
function run_M_24()
    %合成因子并保存
    M_factor_24;
    %验证因子
    M_rule24_update;
end
function run_M_com_2factor()
    M_com_2factor;
end
function run_M_com_2factor_intersect()
    M_com_2factor_intersect;
end
function run_M_com_3factor()
    M_com_3factor;
end
function run_M_com_2factor_feeTest()
    M_com_2factor_feeTest;
end
function run_M_com_2factor_intersect_feeTest()
    M_com_2factor_intersect_feeTest;
end
function run_M_com_3factor_feeTest()
    M_com_3factor_feeTest;
end
function run_M_com_2factor_closetimeTest()
    M_com_2factor_closetimeTest;
end
function run_M_com_2factor_intersect_closetimeTest()
    M_com_2factor_intersect_closetimeTest;
end
function run_M_com_3factor_closetimeTest()
    M_com_3factor_closetimeTest;
end
function run_M_factor_price()
    M_factor_price;
end
function run_M_com_2factor_final()
    M_com_2factor_final;
end
function run_M_com_2factor_intersect_final()
    M_com_2factor_intersect_final;
end
function run_M_com_3factor_final()
    M_com_3factor_final;
end
function run_M_figure_final()
    M_figure_final
end
function run_pause()
    sprintf('按空格键继续！')
    pause
    close all
end
