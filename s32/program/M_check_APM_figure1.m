clear
t = '2016-08-31';

sql_str0 = 'select symbol,f_val from S32.factor_delta0 where tradingdate = ''2016-08-31''';

sql_str1 = 'select symbol,f_val from S32.factor_delta where tradingdate = ''2016-08-31''';

sql_str2 = 'select symbol,f_val from S32.ret20d_update where tradingdate = ''2016-08-31''';

x0 = fetchmysql(sql_str0,2);
x1 = fetchmysql(sql_str1,2);
x2 = fetchmysql(sql_str2,2);

inds = suscc_intersect({x0(:,1),x1(:,1),x2(:,1)});
x0 = cell2mat(x0(inds(:,1),2));
x1 = cell2mat(x1(inds(:,2),2));
y = cell2mat(x2(inds(:,3),2));

figure;
plot(x0,y,'.');
hold on 
plot(x1,y,'.');
legend({'指数0','指数1'})

box off

%apm分布
sql_str = 'select f_val from S32.factor_apm where tradingdate = ''%s''';
f = fetchmysql(sprintf(sql_str,t));
figure
histogram(f,80)
