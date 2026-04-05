%{
多头策略：收盘前半小时（15 分钟）下跌做多，持有至次日上午 10 点平仓。
多空策略：收盘前半小时（15 分钟）下跌做多，收盘前半小时（15 分钟）上涨做
空，持有至次日上午 10 点平仓。
未完成
%}
clear
key_str = 'S28:合成因子3';
f_type = 3;
tn_f = 'S28.comfactors';
var_info = {'symbol','tradingdate','f_type','f_val','f_val2'};

sql_str  =['select tradingdate,t_hour*100+t_minute,price from pytdx_data.%s_tdx_min ',...
    'where price is not null and price>0 and tradingdate>''%s'' order by tradingdate,t_hour,t_minute'];



cut_time = [1445,1500;1430,1500];

sql_str1 = ['select tradingdate from S28.comfactors where f_type=%d and ',...
    'symbol = ''%s'' order by tradingdate desc limit 1'];

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i0 = 1:T_dns
    sub_str = dns{i0};
    t0 = fetchmysql(sprintf(sql_str1,f_type,sub_str),2);
    if isempty(t0)
        t0 = '2016-12-28';
    else
        t0 = t0{1};
    end
    
    sub_str_l = length(sub_str);
    x = fetchmysql(sprintf(sql_str,sub_str,t0),2);
    if isempty(x)
        continue
    end

    tref_all = x(:,1);
    tref = unique(tref_all);
    
    t_min = cell2mat(x(:,2));
    x = cell2mat(x(:,3));
    x=[0;x(2:end)./x(1:end-1)-1];
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
        
        sprintf('%s-%s: %d-%d',key_str,sub_str,i,T)
    end
    
    F = [tref,tref,tref,num2cell([y1,y2])];
    F(:,1) = {sub_str};
    F(:,3) = {f_type};
    %save(sprintf('F21_%s.mat',sub_str),'F')
    datainsert_adair(tn_f,var_info,F)


end