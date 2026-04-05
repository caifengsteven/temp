%{
时间序列动量效应
做多前 R 个交易日上涨的品种，做空前 R 个交易日下跌的品种，每隔 H
个交易日调整一次；
R=H=10

避免起点日期的影响，将初始资金等权分配到 10 个账户中，每个账户初始
日期相差一个交易日，将 10 个账户里的资金汇总计算每日的净值

%}

clear
%close all

R = 10;
H = 10;

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
r_re = y_re;
for symbol_sel = 1:T
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');
    [cash_flow,sub_tref]=get_bac_data(symbol,M(symbol_sel),0.5,100000);

    [~,ia] = intersect(tref_num,sub_tref,'stable');
    y_re(ia,symbol_sel) = [0;cash_flow(2:end)./cash_flow(1:end-1)-1];
    
    [v,sub_tref2] = get_vol_data(symbol);
    [~,ib] = intersect(tref_num,sub_tref2,'stable');
    vol_re(ib,symbol_sel) = movmean(v,[20,0]);
    
    [r,sub_tref3] = get_futurn_return(symbol,R);
    [~,ib] = intersect(tref_num,sub_tref3,'stable');
    r_re(ib,symbol_sel) = r;
    
    
    sprintf('BacTest %d-%d',symbol_sel,T)
end

%com
T_tref = length(tref);
y_bac = zeros(T_tref,H);

ind_ini = find(sum(y_re,2),1);
if ind_ini<H+1
    ind_ini = H+1;
end
for i0=1:H
    for i = ind_ini+i0-1:H:T_tref
        %选定数据
        ind_sel0 = find(~eq(y_re(i,:),0)&vol_re(i,:)>10000);
        sub_r = r_re(i-1,ind_sel0);
        ind_sel1 = ind_sel0(sub_r<0);
        ind_sel2 = ind_sel0(sub_r>0);

        %获取收益率数据,并平均
        sub_ind = i:(i+H-1);
        sub_ind(sub_ind>T_tref) = [];

        if ~isempty(ind_sel2)
            %多
            sub_y_r_m = y_re(sub_ind,ind_sel2);    
            %手续费
            sub_y_r_m(1,:) = sub_y_r_m(1,:)-3/10000;
            sub_y_r_m(end,:) = sub_y_r_m(end,:)-3/10000;
            temp = 1/size(sub_y_r_m,2)*cumprod((1+sub_y_r_m));
            temp = [1;sum(temp,2)];
            temp_m = temp(2:end)./temp(1:end-1)-1;
        else
            temp_m = 0;
        end

        if ~isempty(ind_sel1)
            %空
            sub_y_r = y_re(sub_ind,ind_sel1);    
            %手续费
            sub_y_r([1,end],:) = sub_y_r([1,end],:);
            temp = 1/size(sub_y_r,2)*cumprod((1+sub_y_r));
            temp = [1;sum(temp,2)];
            temp = temp(2:end)./temp(1:end-1)-1;

        else
            temp=0;
        end
        y_bac(sub_ind,i0) = temp_m-temp;
    end
end
y_bac1 = y_bac;
y_bac1(y_bac1>0.1) = 0.1;
y_bac1(y_bac1<-0.1) = -0.1;
y_bac_t = sum(1/i0*cumprod(y_bac1+1),2);
bpcure_plot_updateV2(tref_num,y_bac_t);

function [r,tref] = get_futurn_return(symbol,N)
sql_str = 'select tradingdate,close_price from futuredata.JJ_future_rehabilitation_data where symbol = ''%s'' and tradingdate <= ''2017-03-01''  and tradingdate>=''2005-01-01'' order by tradingdate';
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));

end


function [x,tref] = get_vol_data(symbol)
sql_str = ['select tradingdate,volume from futuredata.price_if_data ',...
        'where variety0=''%s'' and variety=''%s''and open>0 ',...
        'and tradingdate <= ''2017-03-01''  and tradingdate>=''2005-01-01'' order by tradingdate'];
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
x = cell2mat(y_jj(:,2));
tref = datenum(y_jj(:,1));
end



