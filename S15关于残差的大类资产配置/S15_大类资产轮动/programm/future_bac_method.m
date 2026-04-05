function [y_bac,re]=future_bac_method(ini_cash,asure_v,multiplier_v,...
    use_ratio,fee,price_close,signal_v)

%{
ini_cash 初始资金
asure_v 保证金比例×100
multiplier_v 合约乘数
use_ratio 资金使用率
fee 手续费
price_close 买入，卖出价格
signal_v 信号
update
更改手续费计算方式
%}


T = length(price_close);
y = zeros(T+1,1); %结算准备金 1
y(1) = ini_cash; 
y_asure = zeros(size(y)); %风险保证金 2
y_share = zeros(size(y)); %手数 3
y_price = zeros(size(y)); %价格 4
y_fee = zeros(size(y)); %手续费 5
y_force = zeros(size(y));% 强制平仓标记

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
            y_fee(i) = y_price(i)*multiplier_v*y_share(i)*fee;
            %结算准备金
            y(i) = y(i-1)-y_asure(i)-y_fee(i);            
        else
            %平仓
            %核算 平仓盈亏
            y_price(i) = price_close(i-1);
            v_gain = (y_price(i)-y_price(i-1))*multiplier_v*y_share(i-1)*sub_signal1;
            y_fee(i) = y_price(i)*multiplier_v*y_share(i-1)*fee;
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

%信号，结算准备金，买卖价格，手续费，保证金，持有手数，强制平仓信号
re = [signal_v,y,y_price,y_fee,y_asure,y_share,y_force];
y_bac = y+y_asure;

end