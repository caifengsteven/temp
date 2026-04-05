%{
多头策略：收盘前半小时（15 分钟）下跌做多，持有至次日上午 10 点平仓。
多空策略：收盘前半小时（15 分钟）下跌做多，收盘前半小时（15 分钟）上涨做
空，持有至次日上午 10 点平仓。
%}
clear

close all
sql_str  =['select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),',...
    'pct_chg from S28.wind_%s where pct_chg is not null order by tradingdate'];
%设置两个时间点收盘前15分钟，收盘前30分钟
cut_time = [1445,1500;1430,1500];

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i0 = 1:T_dns
    sub_str = dns{i0};
    sub_str_l = length(sub_str);
    x = fetchmysql(sprintf(sql_str,sub_str),2);

    tref_all = x(:,1);
    tref = unique(tref_all);
    
    t_min = cell2mat(x(:,2));
    x = cell2mat(x(:,3));
    
    T = length(tref);
    y1 = zeros(T,1);
    y2 = y1;
    parfor i = 1:T
        sub_ind=strcmp(tref_all,tref(i));
        sub_sub_x = x(sub_ind,:);
        sub_sub_t = t_min(sub_ind,:);
        sub_sub_y = zeros(1,2);
        for j = 1:2
            sub_sub_r = sub_sub_x(sub_sub_t>=cut_time(j,1)&sub_sub_t<=cut_time(j,2));
            temp = cumprod(1+sub_sub_r)-1;
            sub_sub_y(j) = temp(end);
        end
        y1(i) = sub_sub_y(1);
        y2(i) = sub_sub_y(2);
        
        sprintf('%s: %d-%d',sub_str,i,T)
    end
    %时间，收盘前15分钟，收盘前30分钟
    F = [tref,num2cell([y1,y2])];
    save(sprintf('F23_%s.mat',sub_str),'F')


end