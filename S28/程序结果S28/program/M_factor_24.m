%{
基差
收盘前15分钟基差变化
%}
clear

close all
sql_str  =['select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),',...
    'closeprice from S28.wind_%s where pct_chg is not null order by tradingdate'];
cut_time = [1445,1500];

dns = {'IF','IH','IC';'300','50','500'};
T_dns = size(dns,2);
for i0 = 1:T_dns
    sub_str = dns{1,i0};    
    x = fetchmysql(sprintf(sql_str,sub_str),2);
    tref_all = x(:,1);
    tref = unique(tref_all);    
    t_min = cell2mat(x(:,2));
    x = cell2mat(x(:,3));
    
    sub_str2 = dns{2,i0};
    x2 = fetchmysql(sprintf(sql_str,sub_str2),2);
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
        
        sprintf('%s: %d-%d',sub_str,i,T)
    end
    
    F = [tref,num2cell(y)];
    save(sprintf('F24_%s.mat',sub_str),'F')


end