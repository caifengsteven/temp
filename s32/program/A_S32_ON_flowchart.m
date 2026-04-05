clear
scripts_run = {'S32_ON_com_inverse.m';'S32_ON_comQ.m';'S32_ON_comRet20d.m';'S32_ON_com_APM_indicator_20220822.m';...
    'S32_ON_com_APM_stat.m';'S32_ON_com_APM.m';...
    'S32_ON_rankIC.m';'S32_ON_finalindicator.m';'S32_ON_output.m'};
T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S32: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end