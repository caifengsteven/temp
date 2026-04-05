%{
短期股价走势的预测 信息 （2 ）——个股盘中 异动
集合竞价的有效性验证

参数敏感性统计

%}
clear
t1 = '2013-01-01';
t2 = '2017-01-01';
sql_str = ['select distinct date(tradingdate) from futuredata.STK_MKT_QUOTATION ',...
    'where tradingdate >=''%s'' and tradingdate<=''%s'' and filling =0 order by tradingdate'];

tref0 = fetchmysql(sprintf(sql_str,t1,t2),2);

tb_name = 'ycz_result.sta_re20190702_last_30min';
re = [];

for d = 1:5;
%%{
sql_str1 = ['select date(tradingdate),closeprice,r1,r2 from %s where ',...
    'abs(closeprice/precoloseprice-1)<0.1 and r1!=0 and r2!=0  and abs(r2/r1-1)<0.1 ',...
    'and d = %d order by tradingdate'];
%}
% sql_str1 = ['select date(tradingdate),closeprice,r1,r2 from %s where ',...
%     'abs(closeprice/precoloseprice-1)<0.1 and r1!=0 and r2!=0 ',...
%     'and d = %d order by tradingdate'];

x = fetchmysql(sprintf(sql_str1,tb_name,d),2);

tref = unique(x(:,1));
tref_num = datenum(tref);
price = cell2mat(x(:,2:end));

T = length(tref);
y = zeros(T,2);
for i = 1:T
    sub_ind = strcmp(x(:,1),tref(i));
    sub_x = price(sub_ind,:);
    y(i,:) = -mean(sub_x(:,2:3)./sub_x(:,1)-1)-3/1000*2;
end

[~,ia] = intersect(tref0,tref,'stable');

y2 = zeros(length(tref0),2);
y2(ia,:) = y;
figure
subplot(1,2,1);plot(cumprod(1+y2(:,1)),'LineWidth',2);
title('上涨异动信号当日收盘价平仓');
subplot(1,2,2);plot(cumprod(1+y2(:,2)),'LineWidth',2);
title('上涨异动信号次日开盘价平仓');

%触发信号次数，负收益次数，胜率，平均收益率
temp = price(:,3)./price(:,1)-1-0.3/1000;
re = [re;d,size(price,1),sum(temp<0),sum(temp<0)/size(temp,1),mean(temp)*100];
end