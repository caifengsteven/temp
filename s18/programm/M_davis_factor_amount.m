%{
戴维斯双击选股模型的构建

一
4月20,7月20日、10月20日和1月20日作为调仓时点
剔除 ST 和上市不满 1 年的新股。

二
1. 单季净利润的同比增速为正且前期净利润大于 300 万；
2. 上一期单季净利润的同比增速为正；
3. 分别计算当期单季净利润的同比增速和上一期单季净利润的同比增速，并对 2 个增速
环比计算二阶增速，要求二阶增速为正，即 2 个季度加速增长；
4. 上一期单季营收为正。
每个季度对筛选出的样本根据二阶增速降序排列，选择前 25 个样本。
回测参数设置如下：
1. 回测期间：2010 年至 2017 年 5 月 31 日；
2. 交易成本：买入 0.1%，卖出 0.2%；
3. 单只股票权重上限：10%；
4. 基准：中证 500 指数×仓位。

将所有数据更换为国泰安的数据，验证结果

%}

%利润转换为单季度利润
%计算环比

clear

%设置参数
max_symbol_num = 25;
date_cut = [1,4,7,10];
year_select = 2010:2017;
max_date = datenum(2017,5,31);

%获取所有选股时间节点
tref_sel = zeros(length(year_select)*length(date_cut),1);
k = 0;
for i = 1:length(year_select)
    for j = 1:length(date_cut)
        temp_t = datenum(year_select(i),date_cut(j),20);
        k = k + 1;
        tref_sel(k) = temp_t;
    end
end
tref_sel(tref_sel>max_date) = [];
T = length(tref_sel);
%获取交易日历
tref = fetchmysql('select distinct(tradingdate) from futuredata.STK_MKT_BWARDQUOTATION where filling = 0 order by tradingdate;',2);
tref_num = datenum(tref);
%ind = tref_num>=tref_sel(1)&tref_num<=max_date;
%tref=tref(ind);
%tref_num = tref_num(ind);
%载入ST信息数据
sql_str = ['select symbol,tradingdate,amount  from futuredata.STK_MKT_BWARDQUOTATION ',...
    'where filling = 0 and (symbol like ''60%%'' or symbol like ''30%%'' or symbol like ''00%%'' ) ',...
    ' and tradingdate>=''%s'' and tradingdate<=''%s'' order by symbol,tradingdate;'];
for i = 1:T
    t1 = tref_sel(i);
    ind = find(tref_num<=t1,1,'last');
    t1_trade_num = tref_num(ind-20:ind);
    t1_trade = tref(ind-20:ind);
    
    sub_x = fetchmysql(sprintf(sql_str,t1_trade{1},t1_trade{end}),2);
    
    sub_symbols = unique(sub_x(:,1));
    sub_sub_re = zeros(size(sub_symbols));
    for j = 1:length(sub_symbols)
        sub_sub_x1 = sub_x(strcmp(sub_x(:,1),sub_symbols(j)),2:3);
        sub_sub_re(j) = sum(cell2mat(sub_sub_x1(:,end)))/21;
        
    end
    
    
    sprintf('%d-%d',i,T)
end




