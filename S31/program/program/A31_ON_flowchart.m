clear
scripts_run = {'S31_ON_cal_IOPV_min_JK.m';'S31_ON_val_P1_5.m';'S31_ON_val_P2_5.m';...
    'S31_ON_com3_update_stapara20200309.m'};

T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S31: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end