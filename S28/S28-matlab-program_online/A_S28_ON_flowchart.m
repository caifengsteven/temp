scripts_run = {'S28_ON_factor_21_JK.m';'S28_ON_factor_22.m';'S28_ON_factor_23_JK.m';...
    'S28_ON_factor_24_JK.m';'S28_ON_factor_price.m';'S28_ON_rule21_update_JK.m';...
    'S28_ON_rule24_update_JK.m';'S28_ON_rule22_update_JK.m';...
    'S28_ON_com_2factor_final.m';'S28_ON_com_3factor_final_V2.m'};

T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S24: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end
