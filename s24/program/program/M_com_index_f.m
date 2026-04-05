%合成指数
%1 可转债等权指数
%2 正股等权指数

%the last tradingdate of one month
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
sql_str_symbol_pool = 'select tickerBond,tickerEqu from %s where tradedate = ''%s'''; %获取可转债对应正股
sql_str_bond_return = 'select tickerBond,chgPct from %s where tradedate=''%s''';%可转债收益
sql_str_symbol_return = 'select symbol,chgPct from %s where tradedate=''%s''';%正股收益

%为了并行，先计算指数持仓
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

%收益率
y = cell(T,1);
parfor i = 1:T
    symbol_pool = symbol_pool_all{i};
    if isempty(symbol_pool)
        continue
    end
    sub_x_bond = fetchmysql(sprintf(sql_str_bond_return,tn1,tref{i}),2);
    sub_x_symbol = fetchmysql(sprintf(sql_str_symbol_return,tn2,tref{i}),2);
    
    sub_y = zeros(size(symbol_pool));
    [~,ia,ib] = intersect(symbol_pool(:,1),sub_x_bond(:,1));
    sub_y(ia,1) =  cell2mat(sub_x_bond(ib,2));
    
    [~,ia,ib] = intersect(symbol_pool(:,2),sub_x_symbol(:,1));
    sub_y(ia,2) =  cell2mat(sub_x_symbol(ib,2));
    
    %可转债、正股收益保存到sub_y
    y{i} = sub_y';
    
    sprintf('%d-%d',i,T)
end

%每日合成指数
y3= zeros(T,2);
for i = 1:T
    sub_y = y{i};
    if isempty(sub_y)
        continue
    end
    sub_y = sub_y';
    
    %去掉收益率为0的（停牌的，数据缺失的）
    del_ind = eq(sub_y(:,1),0);
    
    sub_y1 = sub_y(:,1);
    sub_y1 = sub_y1(~del_ind);
    sub_y1(isnan(sub_y1)) = [];
    temp_y2 = zeros(1,2);
    %可转债
    if ~isempty(sub_y1)
        temp_y2(1) = mean(sub_y1);
    end
    
    sub_y2 = sub_y(:,2);
    sub_y2 = sub_y2(~del_ind);
    %股票
    if ~isempty(sub_y2)
        temp_y2(2) = mean(sub_y2);
    end
    y3(i,:) = temp_y2;
end

%可视化
figure;
% subplot(2,1,2)
axes( 'YAxisLocation', 'right', 'XAxisLocation', 'top')
%可转债的期权收益 按照6年10%记
r_bond = exp(log(1.1)/244/6)-1;
y3(:,1) = y3(:,1)+r_bond;

plot(tref_num,cumprod(1+y3),'LineWidth',2)
legend({'可转债','正股'},'NumColumns',2);
datetick('x','yyyy')
grid on
y_c = cumprod(1+y3);
[v,v_str,sta_val] = curve_static0(y_c(:,1));

save bac_curve tref tref_num y_c
