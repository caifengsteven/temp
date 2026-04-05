clear
order_strs = {'S13_ON_M_backTest_ETF.m';'S13_ON_backTest_astock.m';...
    'S13_ON_backTest_fushare.m';'S13_ON_backTest_indicator.m'};

for i = 1:length(order_strs)
    order_str = order_strs{i}(1:end-2);
    title_str = sprintf('S13—È÷§-%d',i);
    run_program_adair(order_str,title_str);
end
