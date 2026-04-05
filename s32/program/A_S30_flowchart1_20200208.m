%{
写入表格操作，只可以运行一次，如果需要再次测试，需要将S32中的对应的表删除
%}
clear
% %1 新建表格
% title_str = '新建表格';
% order_str = 'M_S32_create_table';  
% run_program_adair(order_str,title_str);
% %2 合成反转因子
% title_str = '合成reverse因子';
% order_str = 'M_com_inverse';  
% run_program_adair(order_str,title_str);
% 
% %3 合成情绪因子Q
% title_str = '合成情绪因子Q';
% order_str = 'M_com_Q';  
% run_program_adair(order_str,title_str);
% 
% %4 合成ret20因子 M_com_ret20d_update
% title_str = '合成ret20因子';
% order_str = 'M_com_ret20d_update';  
% run_program_adair(order_str,title_str);

%5 APM：构建市场分钟指数 M_com_APM_indicator_update
title_str = 'APM：构建市场分钟指数';
order_str = 'M_com_APM_indicator_update';  
run_program_adair(order_str,title_str);

%6 APM：合成STAT指标  M_com_APM_stat_update
title_str = 'APM：合成STAT指标';
order_str = 'M_com_APM_stat_update';  
run_program_adair(order_str,title_str);

%7 APM：合成APM指标  M_com_APM_stat_update
title_str = 'APM：合成APM指标';
order_str = 'M_com_APM_update';  
run_program_adair(order_str,title_str);

%8 合成综合因子：rankIC
title_str = '合成综合因子：rankIC';
order_str = 'M_com_rankIC';  
run_program_adair(order_str,title_str);

%9 合成综合因子
title_str = '合成综合因子';
order_str = 'M_com_rankIC';  
run_program_adair(order_str,title_str);








