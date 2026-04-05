%合成指数
%1 可转债等权指数
%2 正股等权指数
%3 剔除可交换债
%4 每日更新
%5 20200413 升级
%the last tradingdate of one month
clear
key_str = '合成指数';
ticker2 = fetchmysql(['select distinct tickerBond from yuqerdata.convertiblebond_dayprice ',...
    'where secShortNameBond like ''%转债%'''],2);

tn1 = 'yuqerdata.convertiblebond_dayprice';
tn2 = 'yuqerdata.yq_dayprice';
%tn3 = 'S24.return_data';
y3_0 = fetchmysql('select tradingdate,f_val1,f_val2 from S24.return_data order by tradingdate',2);

%must be deleted
%y3_0 = y3_0(1:end-5,:);

month_cut_date = yq_methods.get_month_end();
tref = yq_methods.get_tradingdate('2006-01-01',datestr(now,'yyyy-mm-dd'));
tref_num = datenum(tref);
t1 = datenum(y3_0(end-35,1));
tref = tref(tref_num>t1);
tref_num = tref_num(tref_num>t1);

[month_cut_date,month_cut_ind] = intersect(tref,month_cut_date,'stable');
T_month_cut = length(month_cut_ind);

T = length(tref);
sql_str_symbol_pool = 'select tickerBond,tickerEqu from %s where tradedate = ''%s'''; %获取可转债对应正股
sql_str_bond_return = 'select tickerBond,chgPct from %s where tradedate=''%s''';%可转债收益
sql_str_symbol_return = 'select symbol,chgPct from %s where tradedate=''%s''';%正股收益

%为了并行，先计算指数持仓
symbol_pool = [];
symbol_pool_all = cell(T,1);
for i = 1:T_month_cut
    sub_ind = month_cut_ind(i);   
    if i < T_month_cut
        sub_ind_n = month_cut_ind(i+1);
    else
        sub_ind_n = T;
    end
    %update symbol pool
    symbol_pool = fetchmysql(sprintf(sql_str_symbol_pool,tn1,tref{sub_ind}),2);
    if ~isempty(symbol_pool)
        [~,ia] = intersect(symbol_pool(:,1),ticker2,'stable');
        symbol_pool = symbol_pool(ia,:);
        symbol_pool_all{sub_ind} = symbol_pool;
    end
    symbol_pool_all(sub_ind+1:sub_ind_n) = symbol_pool_all(sub_ind);
    
    sprintf('%s: %d-%d',key_str,i,T_month_cut)
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
    if isempty(sub_x_bond) || isempty(sub_x_symbol)
        continue
    end  
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
%对接
ia = tref_num>datenum(y3_0(end,1));
y3_new = y3(ia,:);
tref_new = tref(ia,:);
tref_num_new = tref_num(ia,:);
%写入数据库
if ~isempty(y3_new)
    tn = 'S24.return_data';
    var = {'tradingdate','f_val1','f_val2'};
    datainsert_adair(tn,var,[tref_new,num2cell(y3_new)])
end
y3 = [cell2mat(y3_0(:,2:3));y3_new];
tref_num = [datenum(y3_0(:,1));tref_num_new];
tref = [y3_0(:,1);tref_new];
y_c = cumprod(1+y3);

plot(tref_num,y_c,'LineWidth',2)
legend({'可转债','正股'},'NumColumns',2);
datetick('x','yyyy')
grid on
[v,v_str,sta_val] = curve_static0(y_c(:,1));
sprintf('数据更新至日期：%s',tref{end})

