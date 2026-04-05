%{
方法验证部分
这部分的程序每个都可以独立运行，直接将order_str字符内的命令粘贴到matlab命令行
运行即可
%}
clear
%8 合成价值因子验证 2005年开始
title_str = '合成价值因子验证 2005年开始';
order_str = 'M_com_factor_Test';  
run_program_adair(order_str,title_str);

%9 合成价值因子HP滤波前后结果对比
title_str = '合成价值因子HP滤波前后结果对比';
order_str = 'M_com_factor_Test_adj';  
run_program_adair(order_str,title_str);

%10 合成价值因子HP滤波前后结果对比-限制股票池
title_str = '合成价值因子HP滤波前后结果对比-限制股票池';
order_str = 'M_com_factor_Test_adj_indicatorLimit';  
run_program_adair(order_str,title_str);

%11 移动平均滤波前后结果
title_str = '合成价值因子移动平均滤波前后结果对比-限制股票池';
order_str = 'M_com_factor_Test_adj_avg_indicatorLimit';  
run_program_adair(order_str,title_str);
