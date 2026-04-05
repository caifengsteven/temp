%{
动量策略和期限结构数据组合策略
将所有满足条件的期货品种等分成高展
期收益率组和低展期收益率组，选择高展期收益率组中复权主力合约累计收益率排名前
50%的品种构成多头组合，选择低展期收益率组中复权主力合约累计收益率排名后 50%
的品种构成空头组合，我们同样将期限结构类型固定为 TS3，策略主要包括排序期 R 和
持有期 H 两个参数。

仅仅使用了yuqer的展期收益率数据

%}

clear
%close all

fix_fushare_info = containers.Map({'CZCE','SHFE','DCE'},{'XZCE','XSGE','XDCE'});

R = 15;
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
r_re_rollreturn = r_re;
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
    
    [r,sub_tref3] = get_futurn_return(symbol,R);
    [~,ib] = intersect(tref_num,sub_tref3,'stable');
    r_re(ib,symbol_sel) = r;
    
    %展期收益率
    symbol1 = symbol;
    symbol1{1} = fix_fushare_info(symbol1{1});
    [r_rt,sub_tref3_rt] = get_roll_return_yq(symbol1,3);
    [~,ib,ia] = intersect(tref_num,sub_tref3_rt,'stable');
    r_re_rollreturn(ib,symbol_sel) = r_rt(ia);
    
    sprintf('BacTest %d-%d',symbol_sel,T)
end



%com
T_tref = length(tref);
y_bac = zeros(T_tref,1);

ind_ini = find(sum(y_re,2),1);
if ind_ini<2
    ind_ini = 2;
end
for i = ind_ini:H:T_tref
    %选定数据
    ind_sel0 = find(~eq(r_re_rollreturn(i-1,:),0)&vol_re(i-1,:)>10000&r_re_rollreturn(i-1,:)<1000);
    sub_r = r_re_rollreturn(i-1,ind_sel0);
    if length(sub_r)>5
        [~,ia] = sort(sub_r);
        ia1 = ia(1:floor(end*0.5));
        ind_sel1 = ind_sel0(ia1);
        [~,ia1_s] = sort(r_re(i-1,ind_sel1));
        ind_sel1 = ind_sel1(ia1_s(1:ceil(end*0.5)));
        
        ia2 = ia(end-floor(end*0.5)+1:end);
        ind_sel2 = ind_sel0(ia2);
        [~,ia2_s] = sort(r_re(i-1,ind_sel2),'descend');
        ind_sel2 = ind_sel2(ia2_s(1:ceil(end*0.5)));
        
    else
        ind_sel1 = [];
        ind_sel2 = ind_sel0;
        
    end    
    
    %获取收益率数据,并平均
    sub_ind = i:(i+H-1);
    sub_ind(sub_ind>T_tref) = [];
    
    %多
    sub_y_r_m = y_re(sub_ind,ind_sel2);    
    %手续费
    sub_y_r_m(1,:) = sub_y_r_m(1,:)-3/10000;
    sub_y_r_m(end,:) = sub_y_r_m(end,:)-3/10000;
    temp = 1/size(sub_y_r_m,2)*cumprod((1+sub_y_r_m));
    temp = [1;sum(temp,2)];
    temp_m = temp(2:end)./temp(1:end-1)-1;
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
    y_bac(sub_ind) = temp_m-temp;
    
    
    
end


y_bac_t = cumprod(y_bac+1);
bpcure_plot_updateV2(tref_num,y_bac_t);

function [r,tref] = get_futurn_return(symbol,N)
sql_str = 'select tradingdate,close_price from futuredata.JJ_future_rehabilitation_data where symbol = ''%s'' and tradingdate <= ''2017-03-01''  and tradingdate>=''2005-01-01'' order by tradingdate';
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));

end
function [r,tref] = get_roll_return_yq(symbol,N)
sql_str = ['select tradingdate,R1,R2,R3,R4 from futuredata.yuqer_future_rollreturn ',...
    'where exchangeCD = ''%s'' and symbol = ''%s'' and tradingdate <= ''2017-03-01''  ',...
    'and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
tref = datenum(x(:,1));
r = cell2mat(x(:,N+1));

end

function [x,tref] = get_vol_data(symbol)
sql_str = ['select tradingdate,volume from futuredata.price_if_data ',...
        'where variety0=''%s'' and variety=''%s''and open>0 ',...
        'and tradingdate <= ''2017-03-01''  and tradingdate>=''2005-01-01'' order by tradingdate'];
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
x = cell2mat(y_jj(:,2));
tref = datenum(y_jj(:,1));
end



