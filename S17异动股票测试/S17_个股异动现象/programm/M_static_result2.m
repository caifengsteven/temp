%{
收益源于何方
上涨下跌异动信号超额累计收益
%}
clear

tb_name = 'ycz_result.sta_re20190701';

% sql_str1 = ['select r1,r2 from ycz_result.sta_re20190701 where r1>0 and d =5 ',...
%     'and abs(r1)<0.1 and abs(closeprice/precoloseprice-1)<0.1 and abs(r2)<0.1'];

sql_str1 = ['select date(tradingdate),r2 from ycz_result.sta_re20190701 where r1>%0.2f and d =%d ',...
    'and abs(closeprice/precoloseprice-1)<0.1'];

t = 5;
r = 0.03;
x = fetchmysql(sprintf(sql_str1,r,t),2);
tref = unique(x(:,1));
r_a = cell2mat(x(:,2));

T = length(tref);
y = zeros(T,1);
y_fee = y;
for i = 1:T
    ia = strcmp(x(:,1),tref(i));
    y(i) = mean(r_a(ia));
    y_fee(i) = 0.3/100*sum(ia);
    sprintf('%d-%d',i,T); 
end

%%%%%%%%%%%%%%%%%%

sql_str1 = ['select date(tradingdate),r2 from ycz_result.sta_re20190701 where r1<-%0.2f and d =%d ',...
    'and abs(closeprice/precoloseprice-1)<0.1'];

x2 = fetchmysql(sprintf(sql_str1,r,t),2);
tref2 = unique(x2(:,1));
r_a2 = cell2mat(x2(:,2));

T2 = length(tref2);
y2 = zeros(T2,1);
y_fee2 = y2;
for i = 1:T2
    ia = strcmp(x2(:,1),tref2(i));
    y2(i) = mean(r_a2(ia));
    y_fee2(i) = 0.3/100*sum(ia);
    sprintf('%d-%d',i,T2); 
end

figure
subplot(2,1,1);
bar(y);
title(' 阈值3%个股上涨异动信号日收益率序列');
subplot(2,1,2)
bar(y2);
title(' 阈值3%个股下跌异动信号日收益率序列');

figure
subplot(1,2,1);
plot(cumprod(1-y-0.3/100*2));
title('上涨异动信号逐日累计收益')
subplot(1,2,2);
plot(cumprod(1+y2-0.3/100*2));
title('下跌异动信号逐日累计收益')