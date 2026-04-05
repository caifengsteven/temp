
title_str = '转换因子';
order_str = 'M_trans_nrProfitLoss';  
run_program_adair(order_str,title_str);

re = cell(13,1);
re{1} = S26_methods.rule1();
re{2} = S26_methods.rule2();
re{4} = S26_methods.rule4();
re{7} = S26_methods.rule7();
re{8} = S26_methods.rule8();
re{10} = S26_methods.rule10();
%ind = [1,2,4,7,8,10];

re = cellfun(@(x) x',re,'UniformOutput',false);
re = [re{:}]';
[~,ia] = sort(re(:,1));
re = re(ia,:);
[~,ia] = unique(re(:,2),'last');
re = re(ia,:);
[~,ia] = sort(re(:,1));
re = re(ia,:);
re = flipud(re);

gui_result(re,sprintf('S26财务风险股%S',datestr(now,'yyyymmdd')),{'识别日期','代码'});

re = cell2table(re,'VariableNames',{'date','symbol'});
writetable(re,sprintf('S26财务风险股%s.csv',datestr(now,'yyyymmdd')));