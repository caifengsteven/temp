%更新S36数据

scripts_run = {'S33_ON_com_cvar14.m';'S33_ON_com_cvar_month1.m';...
    'S33_ON_com_cvar_month2.m';'S33_ON_com_zxh.m';'S33_ON_output.m'};
T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S33: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end

