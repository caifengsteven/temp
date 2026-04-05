clear
%注意，增加了5d的中间因子数据，需要发送。
scripts_run = {'S22_ON_com_1d_basicfactor.m';'S22_ON_com_1d.m';'S22_ON_com_5d_single.m';...
    'S22_ON_com_5d.m';'S22_ON_com_month.m';'S22_ON_output.m'};
T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S22: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end