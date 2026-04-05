%{
复合信号策略策略
%}

clear
close all

sigma_targ = 0.1;
mod = 5;%3因子还是5因子
R = 30;
H = 10;


sql_str = ['select distinct tradingdate from futuredata.price_if_data where ',...
    'tradingdate <= ''2017-07-31'' and tradingdate>=''2005-01-01''',...
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
r_re1 = y_re;
r_re2 = y_re;
r_re3 = y_re;
r_re4 = y_re;
r_re5 = y_re;
close_price = y_re;
close_price_r = y_re;
for symbol_sel = 1:T
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');
    [cash_flow,sub_tref]=get_bac_dataV2(symbol,M(symbol_sel),0.2,1000000);

    [~,ia] = intersect(tref_num,sub_tref,'stable');
    y_re(ia,symbol_sel) = [0;cash_flow(2:end)./cash_flow(1:end-1)-1];
    
    [v,sub_tref2] = get_vol_data(symbol);
    [~,ib] = intersect(tref_num,sub_tref2,'stable');
    vol_re(ib,symbol_sel) = movmean(v,[20,0]);
    
    [r,sub_tref3,sub_close_price] = get_momentum(symbol,40);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    r_re1(ib,symbol_sel) = r(ia);
    close_price(ib,symbol_sel) = sub_close_price(ia,1);
    close_price_r(ib,symbol_sel) = sub_close_price(ia,2);
    
    [r,sub_tref3] = get_sectional_momentum(symbol,40);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    r_re2(ib,symbol_sel) = r(ia);
    
    [r,sub_tref3] = get_roll_return_yq(symbol,4);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    r_re3(ib,symbol_sel) = r(ia);
    
    [r,sub_tref3] = get_basismomentum_return(symbol,120);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    r_re4(ib,symbol_sel) = r(ia);
    
    [r,sub_tref3] = get_warehouse(symbol,90);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    r_re5(ib,symbol_sel) = -r(ia);
    
    sprintf('BacTest %d-%d',symbol_sel,T)
end

%com
T_tref = length(tref);
m_num = 5;
m_num_2 = floor(H/m_num);
y_bac = zeros(T_tref,m_num);
ind_ini = find(sum(y_re,2),1);
if ind_ini<R
    ind_ini = (R+1);
end
for i0 = 1:m_num
    for i = ind_ini+(i0-1)*m_num_2:H:T_tref
        %1/K
        %选定数据
        temp = cumprod(close_price(i-5:i,:));
        ind_sel0 = find(~eq(y_re(i,:),0)&vol_re(i,:)>10000&~eq(temp(end,:),0));
        
        sub_r2 = r_re2(i-1,ind_sel0);
        sub_r3 = r_re3(i-1,ind_sel0);
        sub_r4 = r_re4(i-1,ind_sel0);
        [~,ia2] = sort(sub_r2);
        [~,ia3] = sort(sub_r3);
        [~,ia4] = sort(sub_r4);
        sub_r5 = r_re5(i-1,ind_sel0);
        [~,ia5] = sort(sub_r5);
        sub_r1 = r_re1(i-1,ind_sel0);
        [~,ia1] = sort(sub_r1);

        k_score = zeros(size(y_re(1,:)));
        for j = 1:mod
            ia = eval(sprintf('ia%d',j));
            if eq(j,1)
                k_score(ia>0) = 1;
                k_score(ia<0) = -1;
            else
                if length(ia)>=5
                    num1 = floor(length(ia)*0.2);
                    ia1 = ia(1:num1);
                    ind_sel1 = ind_sel0(ia1);
                    ia2 = ia(end-num1+1:end);
                    ind_sel2 = ind_sel0(ia2);
                    k_score(ind_sel1) = k_score(ind_sel1)-1;
                    k_score(ind_sel2) = k_score(ind_sel2)+1;
                end
            end            
        end
        
        ind_sel1 = find(k_score<0);
        ind_sel2 = find(k_score>0);
        
        %归一化多、空权重
        sub_w = zeros(size(k_score));
        sub_w(k_score>0) = k_score(k_score>0)./sum(k_score(k_score>0));
        sub_w(k_score<0) = k_score(k_score<0)./sum(k_score(k_score<0));
        if i > 300
            sub_w0 = sub_w;
            %调整权重
            sub_x = y_re(i-240:i,:);
            sigma_s = std(close_price_r(i-240:i,:))./mean(close_price(i-240:i,:));
            %less
            v1 = 0;
            for j = 1:length(ind_sel1)
                sub_sub_x = sub_x(:,ind_sel1);
                sub_v1 = sub_w(ind_sel1(j)).*sub_w(ind_sel1).*corr(sub_sub_x(:,j),sub_sub_x);
                sub_v1 = -sub_v1(j) + sum(sub_v1);
                v1 = v1 + sub_v1;
            end
            sub_w(ind_sel1) = sigma_targ*sub_w(ind_sel1)./(sigma_s(ind_sel1)*sqrt(sum(sub_w(ind_sel1).^2)+2*v1));
            %more
            v2 = 0;
            for j = 1:length(ind_sel2)
                sub_sub_x = sub_x(:,ind_sel2);
                sub_v1 = sub_w(ind_sel2(j)).*sub_w(ind_sel2).*corr(sub_sub_x(:,j),sub_sub_x);
                sub_v1 = -sub_v1(j) + sum(sub_v1);
                v2 = v2 + sub_v1;
            end
            sub_w(ind_sel2) = sigma_targ*sub_w(ind_sel2)./(sigma_s(ind_sel2)*sqrt(sum(sub_w(ind_sel2).^2)+2*v2));
            
            if any(isnan(sub_w(ind_sel1)))
                sub_w(ind_sel1) = sub_w0(ind_sel1);
            end
            if any(isnan(sub_w(ind_sel2)))
                sub_w(ind_sel2) = sub_w0(ind_sel2);
            end
            
            sub_w(ind_sel1) = sub_w(ind_sel1)/sum(sub_w(ind_sel1));
            sub_w(ind_sel2) = sub_w(ind_sel2)/sum(sub_w(ind_sel2));
            
        end
        %获取收益率数据,并平均
        sub_ind = i:(i+H-1);
        sub_ind(sub_ind>T_tref) = [];
        %多
        sub_y_r_m = y_re(sub_ind,ind_sel2);    
        %手续费
        sub_y_r_m(1,:) = sub_y_r_m(1,:)-3/10000;
        sub_y_r_m(end,:) = sub_y_r_m(end,:)-3/10000;
        temp = sub_w(ind_sel2).*cumprod((1+sub_y_r_m));
        temp = [1;sum(temp,2)];
        temp_m = temp(2:end)./temp(1:end-1)-1;
        if ~isempty(ind_sel1)
            %空
            sub_y_r = y_re(sub_ind,ind_sel1);    
            %手续费
            sub_y_r([1,end],:) = sub_y_r([1,end],:);
            temp = sub_w(ind_sel1).*cumprod((1+sub_y_r));
            temp = [1;sum(temp,2)];
            temp = temp(2:end)./temp(1:end-1)-1;
        else
            temp=0;
        end
        y_bac(sub_ind,i0) = temp_m-temp;
    end
end
ind = tref_num>datenum(2010,1,1);
y_bac1 = y_bac(ind,:);
tref_num1 = tref_num(ind);

y_bac_t = 1/m_num*cumprod(y_bac1+1);
y_bac_t = sum(y_bac_t,2);
bpcure_plot_updateV2(tref_num1,y_bac_t);
[v,v_str,sta_val] = curve_static(y_bac_t)

%时间序列动量因子
function [r,tref,add_v] = get_momentum(symbol,N)
sql_str = 'select tradingdate,close_price from futuredata.JJ_future_rehabilitation_data where symbol = ''%s'' and tradingdate <= ''2017-07-31''  and tradingdate>=''2005-01-01'' order by tradingdate';
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));

add_v = [y,[0;y(2:end)./y(1:end-1)]];
end
%截面动量因子
function [r,tref] = get_sectional_momentum(symbol,N)
sql_str = 'select tradingdate,close_price from futuredata.JJ_future_rehabilitation_data where symbol = ''%s'' and tradingdate <= ''2017-07-31''  and tradingdate>=''2005-01-01'' order by tradingdate';
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));
end
%展期收益率因子
function [r,tref] = get_roll_return_yq(symbol,N)
sql_str = ['select tradingdate,R1,R2,R3,R4 from futuredata.yuqer_future_rollreturn ',...
    'where  symbol = ''%s'' and tradingdate <= ''2017-03-01''  ',...
    'and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,symbol{2}),2);
tref = datenum(x(:,1));
r = cell2mat(x(:,N+1));

end
%基差动量
function [r,tref] = get_basismomentum_return(symbol,N,mod)
if nargin < 3
    mod = 1;
end
sql_str = 'select tradingdate,R1,R2,R3,R4,R5 from futuredata.yuqer_future_basis_momentum where symbol = ''%s'' and tradingdate <= ''2017-07-31''  and tradingdate>=''2005-01-01'' order by tradingdate';
x = fetchmysql(sprintf(sql_str,symbol{2}),2);
y = cell2mat(x(:,2:end));
y(y>0.1) = 0.1;
y(y<-0.1) = -0.1;
y = cumprod(1+y);
r = zeros(size(y));
r(N+1:end,:) = y(N+1:end,:)./y(1:end-N,:);%累积收益率
%当月 次月 主力 次主力 最远月
tref = datenum(x(:,1));
if eq(mod,1)
    r = r(:,1)-r(:,3);
elseif eq(mod,2)
    r = r(:,3)-r(:,4);
end

end
%仓单因子
function [r,tref] = get_warehouse(symbol,R)
    %R = 240;
    sql_str = 'select tradedate,wrvol from futuredata.yq_warehousefactor_data where contractobject = ''%s'' and tradedate <= ''2017-07-31''  and tradedate>=''2005-01-01'' order by tradedate';
    x = fetchmysql(sprintf(sql_str,symbol{2}),2);
    if ~isempty(x)
        y = cell2mat(x(:,2:end));
        r = zeros(size(y));
        for i = R+1:length(y)
            r(i) = (y(i)-y(i-R))/y(i-R);
        end
        r(isnan(r)|isinf(r)) = 0;
        tref = datenum(x(:,1));
    else
        r = [];
        tref = [];
    end
end


function [x,tref] = get_vol_data(symbol)
sql_str = ['select tradingdate,volume from futuredata.price_if_data ',...
        'where variety0=''%s'' and variety=''%s''and open>0 ',...
        'and tradingdate <= ''2017-07-31''  and tradingdate>=''2005-01-01'' order by tradingdate'];
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
x = cell2mat(y_jj(:,2));
tref = datenum(y_jj(:,1));
end


