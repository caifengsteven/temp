%{
全多框架

框架缺点
结算准备金小于0时强制部分平仓
%}
clear
symbol = 'I';

obj1= ad_future_method();
sql_str1 = obj1.get_future_basic_info_yq(symbol);
sql_str2 = obj1.get_future_data_yq(symbol);

%上市日期，保证金比例，合约乘数，最小变动单位，最后交易日
x1 = fetchmysql(sql_str1,2);
%asure_v = x1{end,2};
asure_v = 20;%
multiplier_v = x1{end,3};

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

%ini_cash
ini_cash = 10000000;

y = zeros(T+1,1); %结算准备金 1
y(1) = ini_cash; 
y_asure = zeros(size(y)); %风险保证金 2
y_share = zeros(size(y)); %手数 3
y_price = zeros(size(y)); %价格 4
y_fee = zeros(size(y)); %手续费 5
y_force = zeros(size(y));% 强制平仓标记
%use_ratio = 0.2;
use_ratio = asure_v/100;
fee = 3/10000;
%信号转换必须为1-0 0-1 0--1 -1-0，不能跳跃，否者需要手动切换
for i = 2:T+1
    sub_signal1 = signal_v(i-1);
    sub_signal2 = signal_v(i);
    if ~eq(sub_signal2,sub_signal1)
        if eq(sub_signal1,0)            
            %建仓
            %买入价格
            y_price(i) = price_close(i-1);
            %买入手数
            y_share(i) = floor(y(i-1)*use_ratio/(multiplier_v*y_price(i)*asure_v/100));
            %风险保证金
            y_asure(i) = y_price(i)*multiplier_v*y_share(i)*asure_v/100;
            %手续费
            y_fee(i) = y_price(i)*multiplier_v*y_share(i)*asure_v/100*fee;
            %结算准备金
            y(i) = y(i-1)-y_asure(i)-y_fee(i);            
        else
            %平仓
            %核算 平仓盈亏
            y_price(i) = price_close(i-1);
            v_gain = (y_price(i)-y_price(i-1))*multiplier_v*y_share(i-1)*sub_signal1;
            y_fee(i) = abs(v_gain*fee);
            y(i) = y(i-1)+v_gain+y_asure(i-1)-y_fee(i);
            y_asure(i) = 0;
            y_share(i) = 0;           
        end
    else
        if eq(sub_signal2,0)
            %空仓
            y(i) = y(i-1);
            y_price(i) = price_close(i-1);
            y_fee(i) = 0;
            y_asure(i) = y_asure(i-1);
            y_share(i) = y_share(i-1);
        else
            y_price(i) = price_close(i-1);
            v_gain = (y_price(i)-y_price(i-1))*multiplier_v*y_share(i-1)*sub_signal1;
            y_fee(i) = 0;
            y_share(i) = y_share(i-1);
            y_asure(i) = y_price(i)*y_share(i)*multiplier_v*asure_v/100;
            y(i) = y(i-1)+y_asure(i-1)-y_asure(i)+v_gain;            
            %是否需要平仓
            if y(i)< 0
                y_force(i) = 1;
                if y_share(i)>0
                    %平所有仓位
                    y_fee(i) = abs(y_asure(i)*fee);
                    y(i) = y_asure(i)-y_fee(i);
                    y_asure(i) = 0;
                    y_share(i) = 0;  
                else
                    %平仓，策略终止
                    y(i+1:end) = y(i);
                    y_share(i+1:end) = y_share(i);
                    y_asure(i+1:end) = y_asure(i);
                end                
            end
        end        
    end

end

re = [signal_v,y,y_price,y_fee,y_asure,y_share,y_force];
k = y+y_asure;
plot(k);









