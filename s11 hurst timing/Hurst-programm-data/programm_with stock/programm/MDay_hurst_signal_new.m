%计算时变hurst指数
%区分信号多空
%以日来计算
%ref 吴启权《趋势的魅力 —— 基于局部 Hurst 指数的交易策略（1 ）》
%warning('off');
clear
close all


mod_ef = 0; % 0只买多  -1 多空都买


hurst_widow =480; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal = 240; %计算hurst时的窗口参数

%ma_window = 5;
%ema_window = ma_window;

t0 = datenum(1990,12,13);
tt = datenum(2009,11,1);
t_cut = datenum(1997,12,1);


sub_data_info = {'上证综指','深证成指'};
sub_data_info = sub_data_info{2};
x = get_index_data_ycz(sub_data_info,t0,tt);

tref = datenum(x(:,1));
open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));


%r1 = [0;diff(log(close_price))];
r1 = [0;close_price(2:end)./close_price(1:end-1)-1];
%r1_w = [0;close_price_w(2:end)./close_price_w(1:end-1)-1];
%r1_w_log= [0;diff(log(close_price_w))];
%r1_w= 0;


T =length(r1);
hurst_exp0 = nan(T,1);
y0 = nan(T,1);
for i = max(hurst_widow,hurst_widow_cal)+1:T    
    sub_wid = i-hurst_widow+1:i;
    %[y0(i),hurst_exp0(i)]=HurstCompute_update(r1(sub_wid));
    [~,y0(i),hurst_exp0(i)] = hurst_rs_update1(r1(sub_wid),(hurst_widow_cal:-10:10)');
    %[~,y0(i),hurst_exp0(i)] = hurst_rs(r1(sub_wid),4);
end

y_20 = MA(y0,20);
y_40 = MA(y0,60);
y_200 = MA(y0,240);
ind = tref>=t_cut;
sub_tref = tref(ind);
sub_r1 = r1(ind);
sub_close_price = close_price(ind);



y = y_20(ind);
hurst_exp = hurst_exp0(ind);

T = length(y);
ind = ones(T,1);
wid_year = 240;
ind2 = zeros(size(ind));
for i = wid_year:T
    
    if all(y(i-6:i-4)>hurst_exp(i-6:i-4)) && all(y(i-3:i)<hurst_exp(i-3:i)) || (y(i-1)>0.5 &&y(i)<0.5)
        temp = cumprod(1+sub_r1(i-wid_year+1:i));
        if temp(end)>1.05
            if eq(mod_ef,0)
                ind(i) = 0;
            else
                ind(i) = -1;
            end
        else
            ind(i) = 1;
        end
        ind2(i) = 1;
    else
        ind(i) = ind(i-1);
    end
end
ind2 = find(ind2);

yyaxis left
plot(sub_tref,y,'LineWidth',2)
hold on
plot(sub_tref,hurst_exp,'r-','LineWidth',2)
plot(sub_tref,ind,'-','LineWidth',2);
for i = 1:length(ind2)
    line(sub_tref([ind2(i),ind2(i)]),[mod_ef,1],'Color','k','LineWidth',3);
end
yyaxis right
plot(sub_tref,sub_close_price);
my_time_label(gca,sub_tref)
box off


%back_test
r_c = zeros(size(ind));
for i = 2:length(ind)
    r_c(i) = sub_r1(i)*ind(i-1);    
end

figure;
plot(sub_tref,cumprod(1+r_c),'LineWidth',2);
hold on
plot(sub_tref,sub_close_price./sub_close_price(1),'LineWidth',2);
my_time_label(gca,sub_tref)

