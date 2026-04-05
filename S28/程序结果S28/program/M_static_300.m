%验证 300指数日内收益分布
clear
close all
sql_str  ='select hour(tradingdate)*100+minute(tradingdate),chgPct from S28.index300 where chgPct is not null';
x = fetchmysql(sql_str);
t_all = x(:,1);
x = x(:,2);
t = unique(t_all);
T = length(t);
y = zeros(T,1);
for i = 1:T
    sub_ind = eq(t_all,t(i));
    sub_x = x(sub_ind);
    y(i) = mean(sub_x);
    sprintf('%d-%d',i,T)
end

figure;
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
y_index = y;
title('沪深300指数日内收益分布');
setpixelposition(gcf,[430,368,1008,420]);

sql_str  ='select hour(tradingdate)*100+minute(tradingdate),chgPct from S28.future300 where chgPct is not null';
x = fetchmysql(sql_str);
t_all = x(:,1);
x = x(:,2);
t = unique(t_all);
T = length(t);
y = zeros(T,1);
for i = 1:T
    sub_ind = eq(t_all,t(i));
    sub_x = x(sub_ind);
    y(i) = mean(sub_x);
    sprintf('%d-%d',i,T)
end

figure
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
y_future1 = y;
title('沪深300股指期货日内收益分布_历史数据为主')
setpixelposition(gcf,[430,368,1008,420]);

sql_str  ='select hour(tradingdate)*100+minute(tradingdate),chgPct from S28.future300_wind where chgPct is not null';
x = fetchmysql(sql_str);
t_all = x(:,1);
x = x(:,2);
t = unique(t_all);
t = t(t>=930 & t<=1500);
T = length(t);
y = zeros(T,1);
for i = 1:T
    sub_ind = eq(t_all,t(i));
    sub_x = x(sub_ind);
    y(i) = mean(sub_x);
    sprintf('%d-%d',i,T)
end

figure
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
title('沪深300股指期货日内收益分布_wind数据为主')
setpixelposition(gcf,[430,368,1008,420]);

figure;
%sub_y = cumprod(1+[y_index,y_future1])-1;
sub_y = cumsum([y_index,y_future1]);
plot(sub_y,'LineWidth',2)
hold on
plot(sub_y(:,2)-sub_y(:,1),'LineWidth',2);
set(gca,'XTick',floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
legend({'沪深300指数','沪深300期货','基差'},'NumColumns',3,'Location','best')
title('IF日内基差变化')
setpixelposition(gcf,[430,368,1008,420]);

%%%%%%%%%%%%持仓量变化
sql_str  ='select hour(tradingdate)*100+minute(tradingdate),oi from S28.future300 where oi is not null';
x = fetchmysql(sql_str);
t_all = x(:,1);
x = x(:,2);
t = unique(t_all);
T = length(t);
y = zeros(T,1);
for i = 1:T
    sub_ind = eq(t_all,t(i));
    sub_x = x(sub_ind);
    y(i) = mean(sub_x);
    sprintf('%d-%d',i,T)
end

figure
plot(y,'LineWidth',2)
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = t(floor(linspace(1,T,15)));
t_str = num2cell(t_str);
t_str = cellfun(@num2str,t_str,'UniformOutput',false);
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
title('IF日内持仓量变化');
setpixelposition(gcf,[430,368,1008,420]);