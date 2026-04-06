%{
提前需要检查数据，保证所有数据正常
%主流程
%合成IOPV
%信号1-ETF
%信号1-股指期货
%信号2
%信号3
%组合信号

第一部分内容只可以运行一次
%}
clear
%数据检查
title_str = '核查并创建表格';
order_str = 'M_S31_create_table';  
run_program_adair(order_str,title_str);
%合成IOPV分钟数据 
title_str = '核查并创建表格';
order_str = 'M_cal_IOPV_min_final';  
run_program_adair(order_str,title_str);





