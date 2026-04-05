clear
scripts_run = {'S24_ON_autosignal_index.m';'S24_ON_autosignal_1.m';'S24_ON_autosignal_2.m';};
%更新隐含波动率数据
dos('python M_winddata.py')

T = length(scripts_run);
for i = 1:T
    title_str = sprintf('S24: step%d',i);
    order_str = scripts_run{i}(1:end-2);  
    run_program_adair(order_str,title_str);
end