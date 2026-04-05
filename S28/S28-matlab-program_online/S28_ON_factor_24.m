%{
基差
收盘前15分钟基差变化
对接pytdx的分钟数据
%}
clear

key_str = 'S28:合成因子4';
f_type = 4;
tn_f = 'S28.comfactors';
var_info = {'symbol','tradingdate','f_type','f_val'};

sql_str  =['select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),',...
    'closeprice from S28.wind_%s where closeprice is not null and date(tradingdate)>''%s'' order by tradingdate'];

sql_str_pytdx1  =['select tradingdate,t_hour*100+t_minute,price from pytdx_data.%s_tdx_min ',...
    'where price is not null and price>0 and tradingdate>''%s'' order by tradingdate,t_hour,t_minute']; %期货
sql_str_pytdx2  =['select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),',...
    'close from pytdx_data.tdx_min_%s where tradingdate>''%s 23:00'' ',...
    'and close is not null and close>0 order by tradingdate']; %指数

sql_str1 = ['select tradingdate from S28.comfactors where f_type=%d and ',...
    'symbol = ''%s'' order by tradingdate desc limit 1'];

cut_time = [1445,1500];

dns = {'IF','IH','IC';'300','50','500'};
T_dns = size(dns,2);
for i0 = 1:T_dns
    sub_str = dns{1,i0};  
    t0 = fetchmysql(sprintf(sql_str1,f_type,sub_str),2);
    if isempty(t0)
        t0 = '2016-12-28';
    else
        t0 = t0{1};
    end    
    
    x = fetchmysql(sprintf(sql_str_pytdx1,sub_str,t0),2);
    if isempty(x)
        continue
    end
    tref_all = x(:,1);
    tref = unique(tref_all);    
    t_min = cell2mat(x(:,2));
    x = cell2mat(x(:,3));
    
    sub_str2 = dns{2,i0};
    x1 = fetchmysql(sprintf(sql_str,sub_str2,t0),2);  
    x3 = fetchmysql(sprintf(sql_str_pytdx2,sub_str2,t0),2);
    x2 = [x1;x3];
    if isempty(x)
        continue
    end
    
    tref_all2 = x2(:,1);
    tref2 = unique(tref_all2);    
    t_min2 = cell2mat(x2(:,2));
    x2 = cell2mat(x2(:,3));
    
    tref = intersect(tref,tref2);
    
    T = length(tref);
    y = zeros(T,1);
    parfor i = 1:T
        sub_ind=strcmp(tref_all,tref(i));
        sub_sub_x = x(sub_ind,:);
        sub_sub_t = t_min(sub_ind,:);
        
        sub_ind2=strcmp(tref_all2,tref(i));
        sub_sub_x2 = x2(sub_ind2,:);
        sub_sub_t2 = t_min2(sub_ind2,:);
        
        [~,ia,ib] = intersect(sub_sub_t,sub_sub_t2);
        
        sub_sub_t = sub_sub_t(ia);
        sub_sub_x = sub_sub_x(ia)-sub_sub_x2(ib);
        sub_ind_sel = sub_sub_t>=cut_time(1) & sub_sub_t<=cut_time(2);
        sub_sub_t = sub_sub_t(sub_ind_sel);
        sub_sub_x = sub_sub_x(sub_ind_sel);
        
        y(i) = sub_sub_x(end)-sub_sub_x(1);
        sprintf('%s-%s: %d-%d',key_str,sub_str,i,T)
    end
    
    F = [tref,tref,tref,num2cell(y)];
    %save(sprintf('F24_%s.mat',sub_str),'F')
    F(:,1) = {sub_str};
    F(:,3) = {f_type};
    %save(sprintf('F21_%s.mat',sub_str),'F')
    datainsert_adair(tn_f,var_info,F)


end