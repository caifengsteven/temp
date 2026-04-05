%验证
%全部使用wind数据，时间范围201611~今
clear
close all
sql_str  ='select hour(tradingdate)*100+minute(tradingdate),pct_chg from S28.wind_%s_update where pct_chg is not null';

dns = {'IF','300','IH','50','IC','500'};
T_dns = length(dns);
for i0 = 1:T_dns
    sub_str = dns{i0};
    x = fetchmysql(sprintf(sql_str,sub_str));
    t_all = x(:,1);
    x = x(:,2);
    t = unique(t_all);
    %t(t<930|t>1500) = [];
    T = length(t);
    y = zeros(T,1);

    for i = 1:T
        sub_ind = eq(t_all,t(i));
        sub_x = x(sub_ind);
        y(i) = mean(sub_x);
        sprintf('%d-%d',i,T)
    end
    if ~eq(mod(i0,2),0)
        figure;
    end
    fig_num = mod(i0,2);
    if eq(fig_num,0)
        fig_num = 2;
    end
    subplot(1,2,fig_num);
    yyaxis left
    bar(y)
    set(gca,'XTick',floor(linspace(1,T,15)));
    yyaxis right
    %plot(cumprod(1+y),'LineWidth',2)
    plot(1+cumsum(y),'LineWidth',2)
    set(gca,'XTick',floor(linspace(1,T,15)));
    t_str = t(floor(linspace(1,T,15)));
    t_str = num2cell(t_str);
    t_str = cellfun(@num2str,t_str,'UniformOutput',false);
    set(gca,'XTickLabel',t_str);
    set(gca,'XTickLabelRotation',90)
    title(sub_str)
    
    if eq(mod(i0,2),0)
        setpixelposition(gcf,[223,365,1345,420]);
        movegui(gcf,'center');
    end

end