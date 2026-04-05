clear
scripts_run ={'S29_ON_transYTY_.m';'S29_ON_f16.m';'S29_ON_preprocessing.m';...
'S29_ON_com_indicator1_3.m';'S29_ON_com_indicator_4_5.m';'S29_ON_com_ttm.m';...
'S29_ON_pubdate_update.m';'S29_ON_output.m'};
T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S29: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end