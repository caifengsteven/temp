%S23对接程序 matlab部分
scripts_run = {'S23_ON_M_comfactor_fenbi_zscore.m';...
    'S23_ON_M_comfactor_fenbi_month.m';'S23_ON_comfactor_zhubi.m';'S23_ON_output.m'};
T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S23: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end