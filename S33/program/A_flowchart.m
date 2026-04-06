%{
创建表格
计算因子

%1 新建表格
title_str = '新建表格';
order_str = 'M_S32_create_table';  
run_program_adair(order_str,title_str);

由于计算量大，并行时间长，我这里运行过程中，两个因子的运算分别有一次中断
%}
clear

%创建表格 M_S33_create_table
%1 新建表格
title_str = '新建表格';
order_str = 'M_S33_create_table';  
run_program_adair(order_str,title_str);

%合成因子
title_str = '合并 合成因子1-2-日度频率 需要12-24小时，可以大大减少数据载入所消耗时间';
order_str = 'M_com_factor_cvar_f';  
run_program_adair(order_str,title_str);

% %合成因子
% title_str = '合成因子1-日度频率 需要12-24小时';
% order_str = 'M_com_factor_cvar';  
% run_program_adair(order_str,title_str);

%合成因子
title_str = '合成因子1-月度频率';
order_str = 'M_com_factor_cvar_month';  
run_program_adair(order_str,title_str);

% %合成因子
% title_str = '合成因子2-日度频率 需要12-24小时';
% order_str = 'M_com_factor_cvar_V2';  
% run_program_adair(order_str,title_str);

%合成因子
title_str = '合成因子2-月度频率';
order_str = 'M_com_factor_cvar_month_V2';  
run_program_adair(order_str,title_str);

%合成中性化需要的数据 M_comfactor_zxh
title_str = '合成中性化需要的数据';
order_str = 'M_comfactor_zxh';  
run_program_adair(order_str,title_str);

% 综合测试 M_groupanal_V3
title_str = '综合测试 大约需要1-2小时';
order_str = 'M_groupanal_V3';  
run_program_adair(order_str,title_str);