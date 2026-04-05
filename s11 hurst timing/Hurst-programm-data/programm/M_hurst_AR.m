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

%Mdl = arima(2,0,0); 
%[EstMdl,EstParamCov,logL,info]  = estimate(Mdl,r1_w);

%[YF,YMSE]  = ar_pred(r1_w,104);
%计算残差项
order = 5;
m = arx(r1_w, order);
resi_pred = resid([r1_w(1:order);r1_w], m);
resi_pred = resi_pred(order+1:end);
resi_pred  = resi_pred.OutputData;

resi_std = movstd(resi_pred,[104,0]);
bar(resi_pred)
hold on
plot([resi_std,-resi_std]*0.8,'LineWidth',2);
legend({'收益率残差','收益率残差上限','收益率残差下限'})
set(gca,'fontsize',12);
