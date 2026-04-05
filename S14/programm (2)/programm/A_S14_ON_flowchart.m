clear
scripts_run = {'S14_ON_cal_warehouse.m';'S14_ON_cal_roll_return.m';...
    'S14_ON_cal_basis_momentum.m';'S14_ON_cal_allvolume.m';...
    'S14_ON_get_future_dominant_contract_rehabilitation_YQ.m';'S14_ON_signal.m'};
T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S14: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end