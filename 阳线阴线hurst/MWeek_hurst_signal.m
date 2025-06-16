%计算时变hurst指数
%ref 孙旭，基于中国股票市场趋势预测的择时策略研究，兰州大学

clear
close all

hurst_widow =60;%233*2; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal =30;% 233; %计算hurst时的窗口参数

ma_window = 3;
%ma_window = 5;
%ema_window = ma_window;

t0 = datenum(2008,1,3);
tt = datenum(2017,11,1);
t_cut = datenum(2010,1,4);


sub_data_info = {'上证综指','深证成指'};
sub_data_info = sub_data_info{1};
x = get_index_data_ycz(sub_data_info,t0,tt);

tref = datenum(x(:,1));
open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));
[tref,open_price,close_price] = get_week_data(tref,open_price,close_price);

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
    [~,y0(i),hurst_exp0(i)] = hurst_rs_update1(r1(sub_wid),(hurst_widow_cal:-5:5)');
    %[~,y0(i),hurst_exp0(i)] = hurst_rs(r1(sub_wid),233);
end

ind = find(tref>=t_cut);
y = y0;%MA(y0,60);

y_std = movstd(y0,[ma_window,0]);
y_mx = movmean(y0,[ma_window,0]);

lb = y_mx-y_std;
ub = y_mx+y_std;

signal = zeros(size(y0));
signal(1:ind(1)+3) = 1;
signal_position = zeros(size(y0));
mark_value = 1;
for i = ind(1)+ma_window:length(signal)
    temp_return = (close_price(i)-close_price(i-ma_window+1))/close_price(i-ma_window+1);
    sub_wid = i-ma_window+1:i;
    if all(y(sub_wid)<y_mx(sub_wid)) && any(y(sub_wid)<lb(sub_wid)) && eq(mark_value,1) %反转
        if temp_return>0
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
    elseif all(y(sub_wid)>y_mx(sub_wid)) && any(y(sub_wid)>ub(sub_wid)) && eq(mark_value,-1) %趋势
        if temp_return>0 
            if eq(signal(i-1),0)
                signal_position(i) = 1;
                signal(i) = 1;
            else
                signal(i) = signal(i-1);
            end
        else
            if eq(signal(i-1),1)
                signal_position(i) = -1;
                signal(i) = 0;
            else
                signal(i) = signal(i-1);
            end
        end
        mark_value = 0;
    else
        signal(i) = signal(i-1);
        if all(y(sub_wid)>=y_mx(sub_wid))&&eq(mark_value,0)
            mark_value=  1;
        elseif all(y(sub_wid)<=y_mx(sub_wid))&&eq(mark_value,0)
            mark_value = -1;
        end
    end
    
end

sub_ub = ub(ind);
sub_lb = lb(ind);
sub_tref = tref(ind);
y = y(ind);
hurst_exp = y_mx(ind);
sub_close_price = close_price(ind);
signal = signal(ind);
signal_position = signal_position(ind);

signal1 = find(eq(signal_position,-1));
signal2 = find(eq(signal_position,1));

yyaxis left
plot(sub_tref,y,'LineWidth',2)
hold on
plot(sub_tref,hurst_exp,'r-','LineWidth',2)
plot(sub_tref,sub_ub,'g-')
plot(sub_tref,sub_lb,'g-')
plot(sub_tref,signal,'k-','LineWidth',2);
for i = 1:length(signal1)
    line(sub_tref([signal1(i),signal1(i)]),[0,1],'Color','g','LineWidth',3);
end
for i = 1:length(signal2)
    line(sub_tref([signal2(i),signal2(i)]),[0,1],'Color',[0.5,0.18,0.56],'LineWidth',3);
end

yyaxis right
plot(sub_tref,sub_close_price,'LineWidth',3);
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