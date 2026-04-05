%创建表格
clear

tb_all = {'M_create_table_f1','M_create_table_nrProfitLoss'};

T = length(tb_all);
for i = 1:T
    do_method(tb_all{i});
end

function do_method(method_name)
    eval([method_name,';'])
end

