%计算时变hurst指数
%区分信号多空
%以日来计算
%ref Hurst 指数在 A 股市场中的应用
clear
close all

hurst_widow =240;%233*2; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal =120;% 233; %计算hurst时的窗口参数

%ma_window = 5;
%ema_window = ma_window;

t0 = datenum(1997,1,3);
tt = datenum(2009,11,1);
t_cut = datenum(1999,1,4);


sub_data_info = {'上证综指','深证成指'};
sub_data_info = sub_data_info{2};
x = get_index_data_ycz(sub_data_info,t0,tt);

tref = datenum(x(:,1));
open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));


r1 = [0;diff(log(close_price))];
%r1_w = [0;close_price_w(2:end)./close_price_w(1:end-1)-1];
%r1_w_log= [0;diff(log(close_price_w))];
%r1_w= 0;

T =length(r1);
hurst_exp0 = nan(T,1);
y0 = nan(T,1);
for i = max(hurst_widow,hurst_widow_cal)+1:T    
    sub_wid = i-hurst_widow+1:i;
    %[y0(i),hurst_exp0(i)]=HurstCompute_update(r1(sub_wid));
    [~,y0(i),hurst_exp0(i)] = hurst_rs_update1(r1(sub_wid),(hurst_widow_cal:-5:5)');
    %[~,y0(i),hurst_exp0(i)] = hurst_rs(r1(sub_wid),233);
end

ind = find(tref>=t_cut);
y = y0;%MA(y0,60);
signal = zeros(size(y0));
signal(1:ind(1)+20) = 1;
signal_position = zeros(size(y0));
mark_value = 1;
for i = ind(1)+20:length(signal)
    temp_return = (close_price(i)-close_price(i-233+1))/close_price(i-233+1);
    sub_wid = i-4:i;
    if all(y(sub_wid)<hurst_exp0(sub_wid)) && eq(mark_value,1)
        if temp_return>0.05
            if eq(signal(i-1),1)
                signal_position(i) = -1;
                signal(i) = 0;
            else
                signal(i) = signal(i-1);
            end
        else
            if eq(signal(i-1),0)
                signal_position(i) = 1;
                signal(i) = 1;
            else
                signal(i) = signal(i-1);
            end
        end
        mark_value = 0;
    else
        signal(i) = signal(i-1);
        if all(y(sub_wid)>=hurst_exp0(sub_wid))&&eq(mark_value,0)
            mark_value=  1;
        end
    end
    
end


sub_tref = tref(ind);
y = y(ind);
hurst_exp = hurst_exp0(ind);
sub_close_price = close_price(ind);
signal = signal(ind);
signal_position = signal_position(ind);

signal1 = find(eq(signal_position,-1));
signal2 = find(eq(signal_position,1));

yyaxis left
plot(sub_tref,y,'LineWidth',2)
hold on
plot(sub_tref,hurst_exp,'r-','LineWidth',2)
for i = 1:length(signal1)
    line(sub_tref([signal1(i),signal1(i)]),[0,1],'Color','g','LineWidth',3);
end
for i = 1:length(signal2)
    line(sub_tref([signal2(i),signal2(i)]),[0,1],'Color',[0.5,0.18,0.56],'LineWidth',3);
end

yyaxis right
plot(sub_tref,sub_close_price);
my_time_label(gca,sub_tref)
box off

%back_test
r1_a = [0;sub_close_price(2:end)./sub_close_price(1:end-1)-1];
r_c = zeros(size(signal));
for i = 2:length(signal)
    r_c(i) = r1_a(i)*signal(i-1);    
end

figure;
plot(sub_tref,cumprod(1+r_c),'LineWidth',2);
hold on
plot(sub_tref,sub_close_price./sub_close_price(1),'LineWidth',2);
my_time_label(gca,sub_tref)
box off

figure;
yyaxis left
plot(y,'LineWidth',2)
hold on
plot(hurst_exp,'r-','LineWidth',2)
for i = 1:length(signal1)
    line(([signal1(i),signal1(i)]),[0,1],'Color','g','LineWidth',3);
end
for i = 1:length(signal2)
    line(([signal2(i),signal2(i)]),[0,1],'Color',[0.5,0.18,0.56],'LineWidth',3);
end
yyaxis right
plot(sub_close_price);