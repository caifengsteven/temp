%计算时变hurst指数
%warning('off');
clear
close all

check_mod = 2;%1 核实hurst指数；2 核实 移动均线 

hurst_widow = 104; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal = 104; %计算hurst时的窗口参数
data_widow = 1;  %0 所有指数数据，否者自定义

ma_window = 12;
ema_window = ma_window;

delta1 = 0.5;%上限参数
delta2 = 0.5;%下限参数


if eq(data_widow,0)
    t0 = datenum(1000,1,1);
else
    t0 = datenum(2000,1,1);
end
tt = datenum(2017,6,30);
t_cut = datenum(2006,12,29);
sub_data_info = {'上证指数','深证成指'};
sub_data_info = sub_data_info{1};
sql_str = ['select tradingdate,open,close from futuredata.indicator_data ',10,...
    'where symbolname = ''%s'' and tradingdate>= ''%s'' and tradingdate<= ''%s'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,sub_data_info,datestr(t0,'yyyy-mm-dd'),datestr(tt,'yyyy-mm-dd')),2);

tref = datenum(x(:,1));
open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));
[tref_w,open_price_w,close_price_w] = get_week_data(tref,open_price,close_price);

ema_close_price_w = EMA(close_price_w,ema_window); %EMA窗口大小参数未说明
ma_close_price_w = MA(close_price_w,ma_window); %EMA窗口大小参数未说明

r1 = close_price(2:end)./close_price(1:end-1)-1;
r1_w = [0;close_price_w(2:end)./close_price_w(1:end-1)-1];

model_ind = find(tref_w>=t_cut,1);

T =length(r1_w);
hurst_exp = zeros(T,1);
y = zeros(T,1);
for i = hurst_widow+1:T
    if eq(hurst_widow,0)
        sub_wid = 1:i;
        temp_v = 52;
    else
        sub_wid = i-hurst_widow:i;%计算hurst指数窗口参数
        temp_v = hurst_widow;
    end
    
    [~,y(i),hurst_exp(i)] = hurst_rs_update1(r1_w(sub_wid),(3:3:min(hurst_widow_cal,temp_v))');
end

hurst_std = movstd(y,[hurst_widow_cal,0]);
hurst_std(1:model_ind-1) = 0;

%{
%根据所有数据计算std
hurst_std = zeros(T,1);
for i = model_ind:T
    sub_x = y(1:i);
    sub_x(eq(sub_x,0)) = [];
    hurst_std(i) = std(sub_x);
end
%}

h = figure;
setpixelposition(h,[318,464,922,514]);
leg_str = [];
yyaxis left
plot(tref_w(model_ind:T),y(model_ind:T),'-','linewidth',2,'color',[0,0.45,0.74])
leg_str = cat(1,leg_str,{'Hurst指数'});
hold on
plot(tref_w(model_ind:T),hurst_exp(model_ind:T)+hurst_std(model_ind:T)*delta1,'-','linewidth',2,'color',[0.93,0.69,0.13]);
leg_str = cat(1,leg_str,{'上限'});
plot(tref_w(model_ind:T),hurst_exp(model_ind:T)-hurst_std(model_ind:T)*delta2,'-','linewidth',2,'color',[0.6 0.8 1]);
leg_str = cat(1,leg_str,{'下限'});
yyaxis right
plot(tref_w(model_ind:T),close_price_w(model_ind:T),'-','linewidth',2,'color',[0.64,0.08,0.18])
leg_str = cat(1,leg_str,{sub_data_info});
if check_mod>1
    hold on
    plot(tref_w(model_ind:T),ma_close_price_w(model_ind:T),'-','linewidth',2,'color',[0.93,0.69,0.13])
    leg_str = cat(1,leg_str,{sprintf('%s平滑后数据',sub_data_info)});
end
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
datetick('x','yyyymmdd');
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
set(gca,'fontsize',12);
legend(leg_str,'Location','best','NumColumns',length(leg_str));
%hurst_rs_test(r1);




