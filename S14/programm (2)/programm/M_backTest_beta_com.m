%{
价值因子
%}

clear
close all

mod = 2;%1 同比 2 环比

R = 4;
H = 80;


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
r_re = y_re;
for symbol_sel = 1:T
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');
    [cash_flow,sub_tref]=get_bac_dataV2(symbol,M(symbol_sel),0.2,100000);

    [~,ia] = intersect(tref_num,sub_tref,'stable');
    y_re(ia,symbol_sel) = [0;cash_flow(2:end)./cash_flow(1:end-1)-1];
    
    [v,sub_tref2] = get_vol_data(symbol);
    [~,ib] = intersect(tref_num,sub_tref2,'stable');
    vol_re(ib,symbol_sel) = movmean(v,[20,0]);
    
    [r,sub_tref3] = get_beta_return(symbol,R,mod);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    r_re(ib,symbol_sel) = r(ia);
    
    
    sprintf('BacTest %d-%d',symbol_sel,T)
end

%com
T_tref = length(tref);
m_num = 5;
m_num_2 = floor(H/m_num);
y_bac = zeros(T_tref,m_num);
ind_ini = find(sum(y_re,2),1);
if ind_ini<R
    ind_ini = (R+1)*240;
end
for i0 = 1:m_num
    for i = ind_ini+(i0-1)*m_num_2:H:T_tref
        %选定数据
        ind_sel0 = find(~eq(y_re(i,:),0)&vol_re(i,:)>10000&~eq(r_re(i-1),0));
        sub_r = r_re(i-1,ind_sel0);
        if length(sub_r)>5
            [~,ia] = sort(sub_r);
            num1 = floor(length(ia)*0.2);
            ia1 = ia(1:num1);
            ind_sel1 = ind_sel0(ia1);
            ia2 = ia(end-num1+1:end);
            ind_sel2 = ind_sel0(ia2);
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
        y_bac(sub_ind,i0) = temp_m-temp;
    end
end
ind = tref_num>datenum(2011,1,1);
y_bac = y_bac(ind,:);
tref_num = tref_num(ind);
y_bac(isnan(y_bac)|y_bac>0.1|y_bac<-0.1) = 0;
y_bac_t = 1/m_num*cumprod(y_bac+1);
y_bac_t = sum(y_bac_t,2);
bpcure_plot_updateV2(tref_num,y_bac_t);

function [r,tref] = get_beta_return(symbol,R,mod)
if nargin < 3
    mod = 1;
end
mod = mod + 1;
R = R*12;
[~,~,cpi_data] = xlsread('cpi.xlsx');
cpi_data = [datenum(cpi_data(2:end,1)),cell2mat(cpi_data(2:end,3:4))];
%月收益率
%回归
sql_str = 'select tradingdate,close_price from futuredata.JJ_future_rehabilitation_data where symbol = ''%s'' and tradingdate <= ''2017-07-31''  and tradingdate>=''2005-01-01'' order by tradingdate';
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
tref = datenum(x(:,1));
month_num = month(tref);
%月度收益率
ind = [0;find(diff(month_num))];
ind = [ind(1:end-1)+1,ind(2:end)];
T = size(ind,1);
y1 = zeros(T,1);
y2 = zeros(T,1);
for i = 1:T-2
    y1(i)=y(ind(i,2))/y(ind(i,1))-1;
    %时间节点
    %sub_t = datevec(tref(ind(i,1)));
    temp_ind = cpi_data(:,1)>tref(ind(i+2,1))&cpi_data(:,1)<tref(ind(i+2,2));
    %if eq(i,37);keyboard;end
    if any(temp_ind)
        y2(i) = cpi_data(temp_ind,mod);
    end
end

y3 = zeros(T,1);
for i = R+1:T
    p = polyfit(y1(i-R:i),y2(i-R:i),1);
    %p = regress(y2(i-R:i),[ones(size(y2(i-R:i))),y1(i-R:i)]);
    %p(1) = [];
    y3(i) = p(1);
end
%back 
r = zeros(size(tref));
for i = 1:T-2
    r(ind(i+2,1):ind(i+2,2)) = y3(i);
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



