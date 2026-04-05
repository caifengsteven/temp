%{
提前需要检查数据，保证所有数据正常
%主流程
%合成IOPV
%信号1-ETF
%信号1-股指期货
%信号2
%信号3
%组合信号

所有验证内容
%}
clear
%% ETF
title_str = '折溢价信号1 开盘vs.收盘-ETF';
order_str = 'M_val_P1_411';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 日内平均 vs. 收盘-ETF';
order_str = 'M_val_P1_412';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 量价指标-ETF';
order_str = 'M_val_P1_42';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 综合指标-ETF 50、300、500结果-ETF';
order_str = 'M_val_P1_5';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 窗口敏感性研究-ETF';
order_str = 'M_val_P1_para_test';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 手续费研究-ETF';
order_str = 'M_val_P1_fee_test';  
run_program_adair(order_str,title_str);
%% 股指期货
title_str = '折溢价信号1 开盘vs.收盘-股指期货';
order_str = 'M_val_P2_411';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 日内平均 vs. 收盘-股指期货';
order_str = 'M_val_P2_412';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 量价指标-股指期货';
order_str = 'M_val_P2_42';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 综合指标-ETF 50、300、500结果-股指期货';
order_str = 'M_val_P2_5';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 窗口敏感性研究-股指期货';
order_str = 'M_val_P2_para_test';  
run_program_adair(order_str,title_str);

title_str = '折溢价信号1 手续费研究-股指期货';
order_str = 'M_val_P2_fee_test';  
run_program_adair(order_str,title_str);
%% Easy 信号验证-单个
title_str = 'Easy 信号1-折溢价信号';
order_str = 'M_val_11';  
run_program_adair(order_str,title_str);

title_str = 'Easy 信号2-放量上涨';
order_str = 'M_val_12';  
run_program_adair(order_str,title_str);

title_str = 'Easy 信号3-周内效应(周五、周一、周二做多；周三做空)';
order_str = 'M_val_13_sec3';  
run_program_adair(order_str,title_str);

%% Easy 信号验证-组合
title_str = 'Easy 信号-综合放量上涨、周内效应信号';
order_str = 'M_val_com3';  
run_program_adair(order_str,title_str);

title_str = 'Easy 信号-综合三个信号';
order_str = 'M_val_com3_update';  
run_program_adair(order_str,title_str);