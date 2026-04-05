%计算时变hurst指数
%区分信号多空
%更改 LB UB 条件
%使用预测者的指数数据
%warning('off');
clear
close all

%读取数据
data_widow = 1;  %0 所有指数数据，否者自定义

t0 = '1013-01-04';
tt = '2019-12-30';

code_all = {'DCE','SHFE','CZCE','CFFEX'};

type1_sel = 1;%选择商品交易所 1 大商所 2上商所 3郑商所 4 股指或者国债期货
infos = '焦炭';%选择交易类别 类别信息可以见后文的import cm data fun中的定义

[f_name,M_V] = import_cm_data();

type1 = code_all{type1_sel};
type2 = f_name{type1_sel}(infos);

db_name = 'futuredata';
stop_value = [2.5,5]/100;
multi_val = M_V{type1_sel}(infos);

%获取数据
sql_str = ['select tradingdate,open,close from futuredata.price_if_data',10,...
    'where variety0=''%s'' and variety=''%s'' and tradingdate>=''%s''',10,...
    'and tradingdate<=''%s'' order by tradingdate'];
sql_str = sprintf(sql_str, type1,type2,t0,tt);
x = fetchmysql(sql_str,2);
tref = datenum(x(:,1));

open_price = cell2mat(x(:,2));
close_price = cell2mat(x(:,3));
[tref_w,open_price_w,close_price_w] = get_week_data(tref,open_price,close_price);

t0 = tref(1);
tt = tref(end);
t_cut = tref(240);
circle_time = 1;

check_mod = 2;%1 核实hurst指数；2 核实 移动均线 

hurst_widow = 50*circle_time*2; %0 所有可用历史数据，否者窗口数据
hurst_widow_cal = 50*circle_time; %计算hurst时的窗口参数

ma_window = 12;
ema_window = ma_window;

delta1 = 1;%上限参数  对结果影响很大
delta2 = 1;%下限参数  对结果影响很大 

K = 0.5;%平仓resi参数
K1 = 0.5;%平仓resi参数


ema_close_price_w = EMA(close_price_w,ema_window); %EMA窗口大小参数未说明
ma_close_price_w = MA(close_price_w,ma_window); %EMA窗口大小参数未说明

r1 = close_price(2:end)./close_price(1:end-1)-1;
r1_w = [0;close_price_w(2:end)./close_price_w(1:end-1)-1];
%r1_w = close_price_w./open_price_w-1;
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
    tempv = cumprod(1+r1_w(max(i-hurst_widow+1,1):i));
    tempv = tempv(end)-1;
    %反转开仓
    if y(i)<=LBound(i) && abs(resi_pred(i))<K*resi_std(i)&&~eq(signal(i-1),-1)
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
    if y(i) > TBound(i) &&~eq(signal(i-1),1)
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
if ~isempty(p1)
plot(tref_w(p1),y(p1),'^','markerfacecolor','r','MarkerEdgeColor','r','markersize',10)
leg_str = cat(1,leg_str,{'趋势开仓'});
end
if ~isempty(p2)
plot(tref_w(p2),y(p2),'s','markerfacecolor','r','MarkerEdgeColor','r','markersize',10)
leg_str = cat(1,leg_str,{'反转开仓'});
end
yyaxis right
plot(tref_w(model_ind:T),close_price_w(model_ind:T),'-','linewidth',2,'color',[0.64,0.08,0.18])
leg_str = cat(1,leg_str,{infos});
if check_mod>1
    hold on
    plot(tref_w(model_ind:T),ma_close_price_w(model_ind:T),'-','linewidth',2,'color',[0.93,0.69,0.13])
    leg_str = cat(1,leg_str,{sprintf('%s平滑后数据',infos)});
end
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
set(gca,'fontsize',12);
legend(leg_str,'Location','best','NumColumns',length(leg_str));
%hurst_rs_test(r1);
%bac

if type1_sel<4
    cash0 = 100000;
else
    cash0 = 1000000;
end
V = bac_test_CTA_update4(signal1,[open_price_w,close_price_w],multi_val,ones(size(signal1)),cash0,4/10000,zeros(size(signal1))-0.1);
r_c =[0;V(2:end)./V(1:end-1)-1];

figure
plot(tref_w(model_ind:end),cumprod(1+(r_c(model_ind:end))),'LineWidth',2);
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);

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

figure
plot(tref_w(model_ind:end),(cumprod(1+(r_c_a(model_ind:end,:)))-1)*100,'LineWidth',2)
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref_w(model_ind:20:end),'xlim',tref_w([model_ind,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);
legend({'总收益','趋势收益','反转收益','做多收益','做空收益'},'NumColumns',length(leg_str))
ylabel('%')
