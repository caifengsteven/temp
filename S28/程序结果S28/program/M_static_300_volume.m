%验证300指数
clear
close all
%%%%%%%%%%%%持仓量变化
sql_str  ='select hour(tradingdate)*100+minute(tradingdate),oi from S28.wind_IF where oi is not null';
x = fetchmysql(sql_str);
t_all = x(:,1);
x = x(:,2);
t = unique(t_all);
t =t(t>=930&t<=1500);
T = length(t);
y = zeros(T,1);
for i = 1:T
    sub_ind = eq(t_all,t(i));
    sub_x = x(sub_ind);
    y(i) = mean(sub_x);
    sprintf('%d-%d',i,T)
end

figure
plot(y,'LineWidth',3)
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = t(floor(linspace(1,T,15)));
t_str = num2cell(t_str);
t_str = cellfun(@num2str,t_str,'UniformOutput',false);
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[430,368,1008,420]);