%计算时变hurst指数
%warning('off');
clear
close all

check_mod = 2;%1 核实hurst指数；2 核实 移动均线 

hurst_widow = 52*1; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal = 52*1; %计算hurst时的窗口参数
data_widow = 1;  %0 所有指数数据，否者自定义

ma_window = 12;
ema_window = ma_window;

delta1 = 1;%上限参数
delta2 = 1;%下限参数

K = 0.8;%resi参数
K1 = 0.8;

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

TBound = hurst_exp+hurst_std*delta1;
LBound = hurst_exp-hurst_std*delta2;

%AR计算残差项
order = 5;
m = arx(r1_w, order);
resi_pred = resid([r1_w(1:order);r1_w], m);
resi_pred = resi_pred(order+1:end);
resi_pred  = resi_pred.OutputData;
resi_std = movstd(resi_pred,[hurst_widow_cal,0]);


signal = zeros(size(hurst_std));
for i = model_ind:length(signal)
    if y(i)<=LBound(i) && abs(resi_pred(i))<K*resi_std(i)
        signal(i) = -1;
        continue
    end
    
    if eq(signal(i-1),-1) && y(i)>hurst_exp(i)
        signal(i) = 0;
        continue
    end
        
    if y(i) > TBound(i)
        signal(i) = 1;
        continue
    end
    
    if eq(signal(i-1),1)
        
        if y(i)<hurst_exp(i) || ...
                (r1_w(i)<0&&y(i)>TBound(i) && resi_pred(i)>K1*resi_std(i)) ||...
                (r1_w(i)>0&&y(i)>TBound(i) && resi_pred(i)<-K1*resi_std(i))
            signal(i) = 0;
        end
        continue        
    end
    
    signal(i) = signal(i-1);
    
end

ind = [0;find(diff(signal));length(signal)];
ind = [ind(1:end-1)+1,ind(2:end)];
p1 = ind((eq(signal(ind(:,1)),1)),1);
p2 = ind((eq(signal(ind(:,1)),-1)),1);
h = figure;
setpixelposition(h,[318,464,922,514]);
leg_str = [];
yyaxis left
plot(tref_w(model_ind:T),y(model_ind:T),'-','linewidth',2,'color',[0,0.45,0.74])
leg_str = cat(1,leg_str,{'Hurst指数'});
hold on
plot(tref_w(model_ind:T),TBound(model_ind:T),'-','linewidth',2,'color',[0.93,0.69,0.13]);
leg_str = cat(1,leg_str,{'上限'});
plot(tref_w(model_ind:T),LBound(model_ind:T),'-','linewidth',2,'color',[0.6 0.8 1]);
leg_str = cat(1,leg_str,{'下限'});
plot(tref_w(p1),y(p1),'^','markerfacecolor','r','MarkerEdgeColor','r','markersize',10)
leg_str = cat(1,leg_str,{'趋势开仓'});
plot(tref_w(p2),y(p2),'s','markerfacecolor','r','MarkerEdgeColor','r','markersize',10)
leg_str = cat(1,leg_str,{'反转开仓'});
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
%bac
r_c = zeros(size(signal));
for i = 2:length(signal)
    r_c(i) = r1_w(i)*signal(i-1);    
end

figure
plot(tref_w(model_ind:20:end),cumprod(1+r_c(model_ind:20:end)),'LineWidth',2);
set(gca,'XTickLabelRotation',90);
datetick('x','yyyymmdd');
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
set(gca,'fontsize',12);







