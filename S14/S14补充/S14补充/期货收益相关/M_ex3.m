%{
全多框架

框架缺点
结算准备金小于0时强制部分平仓
在ex2的基础上将各部分模块化
%}
clear
symbol = 'I';

obj1= ad_future_method();
sql_str1 = obj1.get_future_basic_info_yq(symbol);
sql_str2 = obj1.get_future_data_yq(symbol);

%参数 

%上市日期，保证金比例，合约乘数，最小变动单位，最后交易日
x1 = fetchmysql(sql_str1,2);
%asure_v = x1{end,2};
asure_v = 20;%保证金比例
multiplier_v = x1{end,3};
ini_cash = 10000000; %ini_cash
%use_ratio = 0.2;
use_ratio = asure_v/100; %建仓资金比例
fee = 3/10000; %手续费


%%交易日期，合约代码，开，手，结
x2 = fetchmysql(sql_str2,2);
price_open = cell2mat(x2(:,3));
price_close = cell2mat(x2(:,4));
price_sta = cell2mat(x2(:,5));

%信号
T = size(x2,1);
signal_v = ones(T,1);
for i = 1:T-1
    if ~strcmp(x2(i,2),x2(i+1,2))
        signal_v(i) = 0;
    end
end
signal_v = [0;signal_v];
signal_v(end) = 0;
%signal_v = -signal_v;
[y_bac,re]=future_bac_method(ini_cash,asure_v,multiplier_v,use_ratio,fee,price_close,signal_v);
plot(y_bac);








