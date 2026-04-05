%{
S29项目流程
我们分步骤将计算结果验证分解

每个步骤结果计算完毕后，在matlab界面按空格继续下一步的验证。

S29.factor_wind_com_ttm S29.factor_wind_com两个表是我们根据基础表合成的
%}
clear
%验证基础数据是否完备
tns_all = {'S29.factor_wind','yuqerdata.MktEqumAdjAfGet','yuqerdata.st_info',...
    'gta_web.gta_idx_smprat','yuqerdata.yq_index_month','yuqerdata.yq_tradingdate',...
    'yuqerdata.yq_industry'};
sql_test = 'select * from %s limit 1';
test_num = nan(size(tns_all));
for i = 1:length(tns_all)
    temp = fetchmysql(sprintf(sql_test,tns_all{i}),2);
    if ~isempty(temp)
        test_num(i) = 1;
    end    
end
tns_miss = tns_all(isempty(test_num));
if ~isempty(tns_miss)
    for i = 1:length(tns_miss)
        sprintf('数据表缺少:%s',tns_miss{i})
    end
else
    sprintf('数据齐备')
end

sprintf('按空格键继续！')
pause
    
%创建表格
title_str = '创建表格';
order_str = 'M_S29_create_table_savere';  
run_pause(order_str,title_str);

%因子数据预处理
title_str = '因子数据预处理';
order_str = 'M_preprocessing';  
run_pause(order_str,title_str);

%合成组合因子
title_str = '合成组合因子1-3';
order_str = 'M_com_indicator';  
run_pause(order_str,title_str);

title_str = '将组合因子4-5写入表';
order_str = 'M_com_indicator_45';  
run_pause(order_str,title_str);

%合成ttm数据
title_str = '合成ttm数据';
order_str = 'M_com_ttm';  
run_pause(order_str,title_str);

%结果验证部分
title_str = '投机性收益反转属性验证';
order_str = 'M_221';   %M_221表示程序文件名称为M_221.m，以下含义相同
run_pause(order_str,title_str);


tilte_str = '基本面收益的动量属性验证';
order_str = 'M_221_update';
run_pause(order_str,title_str);

title_str = '双向选择效果验证';
order_str =  'M_223';
run_pause(order_str,title_str);

title_str = '根据文献描述验证结果';
order_str = 'M_31_update2';
run_pause(order_str,title_str);

title_str = ['限定股票池为500，但是选股时，行业内按照行业内排名，取趋势和反转属性',...
    '的前10%的股票的交集作为股票池（若选不到，则从20%股票池选，若还选不到则从30%选，直至选出股票）'];
order_str = 'M_31_update3';
run_pause(order_str,title_str);

title_str = ['限定股票池为500，但是选股时，不在行业内排名，而是整体排名，取趋势',...
    '和反转属性的前10%的股票的交集作为股票池（若选不到，则从20%股票池选，若还选不到则从30%选，直至选出股票）'];
order_str = 'M_31_update1';
run_pause(order_str,title_str);

title_str = ['不限定股票池，且选股时，不在行业内排名，而是整体排名，取趋势和反转',...
    '属性的前10%的股票的交集作为股票池（若选不到，则从20%股票池选，若还选不到则从30%选，直至选出股票）'];
order_str = 'M_31';
run_pause(order_str,title_str);


function run_pause(order_str,title_str)
    sprintf('%s',title_str)
    eval(order_str)
    sprintf('按空格键继续！')
    pause
    close all
end