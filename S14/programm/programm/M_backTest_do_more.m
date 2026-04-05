%{
首先考虑构建一个等权做多策略作为基准，即等权买入所有满足条件（上市时间、
流动性）的商品期货主力合约，每隔 H（20）个交易日再平衡。

上市满半年
前N日交易量均值大于1万手
%}

%合约乘数
%以螺纹钢为例子
clear
close all

sql_str = ['select distinct tradingdate from futuredata.price_if_data where ',...
    'tradingdate <= ''2017-03-01'' and tradingdate>=''2005-01-01''',...
    ' order by tradingdate'];
tref = fetchmysql(sql_str,2);
tref_num = datenum(tref);

%list
[~,~,x] = xlsread('future_info.xlsx','sheet3');
symbol0 = cellfun(@(x,y) [x,'.',y],x(:,1),x(:,2),'UniformOutput',false);
sy_info0 = x(:,3);
M = cell2mat(x(:,6));
T = length(symbol0);

y_re = zeros(length(tref),T);
vol_re = y_re;
for symbol_sel = 1:T
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');
    [cash_flow,sub_tref]=get_bac_data(symbol,M(symbol_sel));

    [~,ia] = intersect(tref_num,sub_tref,'stable');
    y_re(ia,symbol_sel) = [0;cash_flow(2:end)./cash_flow(1:end-1)-1];
    
    [v,sub_tref2] = get_vol_data(symbol);
    [~,ib] = intersect(tref_num,sub_tref2,'stable');
    vol_re(ib,symbol_sel) = movmean(v,[20,0]);
    
    sprintf('BacTest %d-%d',symbol_sel,T)
end

%com
T_tref = length(tref);
y_bac = zeros(T_tref,1);
N = 20;

ind_ini = find(sum(y_re,2),1);

for i = ind_ini:N:T_tref
    %选定数据
    %ind_sel = ~eq(y_re(i,:),0)&vol_re(i,:)>10000;
    ind_sel = ~eq(y_re(i,:),0);
    %获取收益率数据,并平均
    sub_ind = i:(i+N-1);
    sub_ind(sub_ind>T_tref) = [];
    sub_y_r = y_re(sub_ind,ind_sel);
    y_bac(sub_ind) = mean(sub_y_r,2);    
end


y_bac_t = cumprod(y_bac+1);
bpcure_plot_updateV2(tref_num,y_bac_t);

function [x,tref] = get_vol_data(symbol)
sql_str = ['select tradingdate,volume from futuredata.price_if_data ',...
        'where variety0=''%s'' and variety=''%s''and open>0 ',...
        'and tradingdate <= ''2017-03-01''  and tradingdate>=''2005-01-01'' order by tradingdate'];
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
x = cell2mat(y_jj(:,2));
tref = datenum(y_jj(:,1));
end



