%{
F、IH、IC 在隔夜、前收盘至次日上午 10 点、下午开盘后五分钟三个不同时段的
区间累计收益率如下图所示
%}
clear

close all
sql_str  ='select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),pct_chg from S28.wind_%s where pct_chg is not null';
cut_time = [930,930;930,1000;1300,1305];
cut_str = {'隔夜','前收盘-10点','下午开盘后五分钟'};

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i0 = 1:T_dns
    sub_str = dns{i0};
    x = fetchmysql(sprintf(sql_str,sub_str),2);
    tref_all = x(:,1);
    tref = unique(tref_all);
    t_min = cell2mat(x(:,2));
    x = cell2mat(x(:,3));
    
    T = length(tref);
    y = zeros(T,3);

    parfor i = 1:T
        sub_ind = strcmp(tref_all,tref(i));        
        sub_x = x(sub_ind);
        sub_t = t_min(sub_ind);
        
        for j = 1:3
            sub_sub_ind = sub_t>=cut_time(j,1) &sub_t<=cut_time(j,2);
            temp = cumprod(1+sub_x(sub_sub_ind))-1;
            y(i,j) = temp(end);
        end
        
        sprintf('%d-%d',i,T)
    end
    y_re = cumprod(1+y)-1;
    figure
    plot(y_re*100,'LineWidth',2);
    legend(cut_str,'NumColumns',3,'Location','northwest')
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    t_str = tref(floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str);
    set(gca,'XTickLabelRotation',90)
    title(sub_str)
    setpixelposition(gcf,[223,365,1345,420]);
    box off

end