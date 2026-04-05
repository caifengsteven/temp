%计算时变hurst指数
%区分信号多空
%更改 LB UB 条件
%使用预测者的指数数据
%warning('off');
%使用过去一段时间累计收益判断多空
%Hurst择时-回测-update20190512
clear
key_str = '移动窗口';

%读取数据
%index_name_pool = {'上证综指','深证成指','上证50','沪深300','中证500','中小板指'};
index_name_pool = {'上证综指','上证50','沪深300','中证500'};
T_index = length(index_name_pool);
t_para = {[datenum(2000,1,1);datenum(2019,6,30);datenum(2006,12,29)],...
    [datenum(2000,1,1);datenum(2019,6,30);datenum(2006,12,29)],...
    [datenum(2000,1,1);datenum(2019,6,30);datenum(2012,12,29)],...
    [datenum(2000,1,1);datenum(2019,6,30);datenum(2013,12,29)],...
    [datenum(2000,1,1);datenum(2019,6,30);datenum(2013,12,29)]};

for index_sel = 1:T_index %选择指数
sub_data_info = index_name_pool{index_sel};

t_cut = t_para{index_sel}(3);
circle_time = [1,1,1,1,0.5]; %周期设置）%1 1 2 2 0.5

x = get_index_data_yuqer(sub_data_info);


check_mod = 1;%1 核实hurst指数；2 核实 移动均线 

hurst_widow = 52*4; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal = 52*circle_time(index_sel); %计算hurst时的窗口参数

ma_window = 12;
ema_window = ma_window;

delta1 = 0.5;%上限参数
delta2 = 0.5;%下限参数

K = 0.5;%平仓resi参数
K1 = 0.5;%平仓resi参数

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
    
    [~,y(i),hurst_exp(i)] = hurst_rs_update1(r1_w(sub_wid),(min(hurst_widow_cal,temp_v):-4:4)');
end
y = MA(y,12);
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
resi_std = movstd(resi_pred,[hurst_widow,0]);


signal = zeros(size(hurst_std));
signal1 = signal;
for i = model_ind:length(signal)
    tempv = cumprod(1+r1_w(max(i-52*1+1,1):i));
    tempv = tempv(end)-1;
    %反转开仓
    if y(i)<=LBound(i) && abs(resi_pred(i))<K*resi_std(i)
        signal(i) = -1;
        if tempv>0.1
            signal1(i) = -1;
        else
            signal1(i) = 1;
        end
        continue
    end
    
    if eq(signal(i-1),-1) && y(i)>hurst_exp(i)
        signal(i) = 0;
        signal1(i) = 0;
        continue
    end
    %趋势开仓    
    if y(i) > TBound(i)
        signal(i) = 1;
        if tempv>0.1
            signal1(i) = 1;
        else
            signal1(i) = -1;
        end
        continue
    end
    
    if eq(signal(i-1),1)
        
        if y(i)<hurst_exp(i) || ...
                (tempv<0&&y(i)>TBound(i) && resi_pred(i)>K1*resi_std(i)) ||...
                (tempv>0&&y(i)>TBound(i) && resi_pred(i)<-K1*resi_std(i))
            signal(i) = 0;
            signal1(i) = 0;
        end
        continue        
    end
    
    signal1(i) = signal1(i-1);
    signal(i) = signal(i-1);
    
end

ind = [0;find(diff(signal));length(signal)];
ind = [ind(1:end-1)+1,ind(2:end)];
p1 = ind((eq(signal(ind(:,1)),1)),1);
p2 = ind((eq(signal(ind(:,1)),-1)),1);

r_c = zeros(size(signal1));
for i = 2:length(signal1)
    r_c(i) = r1_w(i)*signal1(i-1);    
end


ind = false(length(signal),5);
ind(:,1) = ~ind(:,1);
ind(:,2) = eq(signal,1);
ind(:,3) = eq(signal,-1);
ind(:,4) = eq(signal1,1);
ind(:,5) = eq(signal1,-1);

r_c_a = zeros(size(ind));
for i = 1:5
    r_c_a(ind(:,i),i) = r_c(ind(:,i));
end
%subplot(T_index,1,index_sel)
figure
plot(tref_w(model_ind:end),(cumprod(1+(r_c_a(model_ind:end,:)))-1)*100,'LineWidth',2)
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);
leg_str2=  {'总收益','趋势收益','反转收益','做多收益','做空收益'};
legend(leg_str2,'NumColumns',length(leg_str2))
title_str = sprintf('%s-%s',key_str,sub_data_info);
title(title_str)
end
%h.Name = key_str;
