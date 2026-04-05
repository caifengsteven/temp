%com index
%1 convert bond index
%2 ticker index
% update
% return:only consider the exist

%the last tradingdate of one month
%价格不复权

clear

tn1 = 'yuqerdata.convertiblebond_dayprice';
tn2 = 'yuqerdata.yq_dayprice';

tref = fetchmysql(sprintf('select distinct tradedate from %s where tradedate <= ''2019-07-31'' order by tradedate',tn1),2);
tref_num = datenum(tref);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
T_month_cut = size(month_cut,1);
month_cut_date = tref(month_cut(:,2));

T = length(tref);
sql_str_symbol_pool = 'select tickerBond,tickerEqu from %s where tradedate = ''%s''';
sql_str_bond_return = 'select tickerBond,chgPct from %s where tradedate=''%s''';
sql_str_symbol_return = 'select symbol,closeprice/actprecloseprice-1 from %s where tradedate=''%s''';

%for parall
symbol_pool = [];
symbol_pool_all = cell(T,1);
for i = 1:T
    if any(strcmp(month_cut_date,tref(i)))
        %update symbol pool
        symbol_pool = fetchmysql(sprintf(sql_str_symbol_pool,tn1,tref{i}),2);
    end
    symbol_pool_all{i} = symbol_pool;
    sprintf('%d-%d',i,T)
end

y = cell(T,1);
parfor i = 1:T
    symbol_pool = symbol_pool_all{i};
    if isempty(symbol_pool)
        continue
    end
    sub_x_bond = fetchmysql(sprintf(sql_str_bond_return,tn1,tref{i}),2);
    sub_x_symbol = fetchmysql(sprintf(sql_str_symbol_return,tn2,tref{i}),2);
    
    sub_y = zeros(2,1);
    [~,~,ib] = intersect(symbol_pool(:,1),sub_x_bond(:,1));
    temp1 = cell2mat(sub_x_bond(ib,2));
    temp1(isnan(temp1)) = [];
    sub_y(1) = mean(temp1);
    
    [~,~,ib] = intersect(symbol_pool(:,2),sub_x_symbol(:,1));
    temp2 = cell2mat(sub_x_symbol(ib,2));
    temp2(isnan(temp2)) = [];
    sub_y(2) = mean(temp2);
    
    y{i} = sub_y';
    
    sprintf('%d-%d',i,T)
end



