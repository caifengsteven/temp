%{
！！！！！
写入表格操作，只可以运行一次，如果需要再次测试，需要将S30中的对应的表删除
%}
clear
%1验证HP方法正确性
title_str = '验证HP方法正确性';
order_str = 'M_figure3';  
run_program_adair(order_str,title_str);

%2创建表格
title_str = '创建表格';
order_str = 'M_S30_create_table';
run_program_adair(order_str,title_str);

%3准备合成的单个因子数据
for i=1:5
    title_str = sprintf('合成 子 因子%d',i);
    order_str = sprintf('M_com_subfactor%d',i);
    run_program_adair(order_str,title_str);
end

%4合成因子
for i=1:5
    title_str = sprintf('合成细分因子%d',i);
    order_str = sprintf('M_com_factor%d',i);
    run_program_adair(order_str,title_str);
end

%5合成价值风格因子
title_str = '合成价值风格因子';
order_str = 'M_com_factor_final';
run_program_adair(order_str,title_str);

%6价值风格因子HP滤波
title_str = '价值风格因子HP滤波';
order_str = 'M_com_factor_final_adj';
run_program_adair(order_str,title_str);

%7价值风格因子移动平均滤波
title_str = '价值风格因子移动平均滤波';
order_str = 'M_com_factor_final_average';
run_program_adair(order_str,title_str);


