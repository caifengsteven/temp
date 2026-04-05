%验证S39 动量及加速策略主程序
%1 
title_str = '月度框架';
order_str = 'M_dongliangjiasu_bac';  
run_program_adair(order_str,title_str);

%2
title_str = '周度框架';
order_str = 'M_dongliangjiasu_week_bac';  
run_program_adair(order_str,title_str);

%{
%3 计算很慢，效果不如周框架，下位测试代码
title_str = '分钟框架';
order_str = 'M_dongliangjiasu_5min_bac';  
run_program_adair(order_str,title_str);
%}