%{
方法验证部分
这部分的程序每个都可以独立运行，直接将order_str字符内的命令粘贴到matlab命令行
运行即可
%}
clear
%1 验证反转因子 rankIC结果
title_str = '验证反转因子rankIC结果';
order_str = 'M_rankIC_reverse';  
run_program_adair(order_str,title_str);

%2 验证反转因子分组回测结果 
title_str = '验证反转因子分组回测结果';
order_str = 'M_group_anal_Neutralization_day_reverse_update';  
run_program_adair(order_str,title_str);

%3 验证Q因子 rankIC结果
title_str = '验证Q因子rankIC结果';
order_str = 'M_rankIC_Q';  
run_program_adair(order_str,title_str);

%4 验证Q因子分组回测结果 
title_str = '验证Q因子分组回测结果';
order_str = 'M_group_anal_Neutralization_day_Q_update';  
run_program_adair(order_str,title_str);

%5 验证APM因子 rankIC结果
title_str = '验证APM因子 rankIC结果';
order_str = 'M_rankIC_APM';  
run_program_adair(order_str,title_str);

%6 验证APM因子分组回测结果 
title_str = '验证APM因子分组回测结果';
order_str = 'M_group_anal_Neutralization_day_APM_update';  
run_program_adair(order_str,title_str);

%7 验证综合因子 rankIC结果
title_str = '验证综合因子 rankIC结果';
order_str = 'M_rankIC_com';  
run_program_adair(order_str,title_str);

%8 验证综合因子分组回测结果 
title_str = '验证综合因子分组回测结果';
order_str = 'M_group_anal_Neutralization_day_comfactor_update';  
run_program_adair(order_str,title_str);

%9 验证不同股票池分组回测结果
title_str = '验证不同股票池分组回测结果';
order_str = 'M_group_pool_anal'; 
run_program_adair(order_str,title_str);