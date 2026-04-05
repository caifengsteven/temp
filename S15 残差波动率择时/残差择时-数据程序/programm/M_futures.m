clear
close all
t0 = '2013-01-04';
tt = '2016-12-30';

warning('off');
db_name = 'futuredata';
type1 = {'SHFE','SHFE','DCE','DCE'};
type2 = {'RU','RB','JM','J'};
infos = {'天然橡胶','螺纹钢','焦煤','焦炭'};
multi_val = [10,10,60,100];
sel = 1;
window =20;
window2 =5;

stop_value = [2.5,1.5]/100;

type1 = type1{sel};
type2 = type2{sel};
infos = infos{sel};
multi_val = multi_val(sel);


cash0=1000000;
%螺纹钢 'RB'

%获取数据
sql_str = ['select tradingdate,open,close from futuredata.price_if_data',10,...
    'where variety0=''%s'' and variety=''%s'' and tradingdate>=''%s''',10,...
    'and tradingdate<=''%s'' order by tradingdate'];
sql_str = sprintf(sql_str, type1,type2,t0,tt);

x = fetchmysql(sql_str,2);
%time
tref_str = x(:,1);
tref = datenum(x(:,1));
%price
P_o = cell2mat(x(:,2));
P = cell2mat(x(:,3));
%计算指标
%残差波动率
P_ema = EMA(P,window);
delta = log(P)-log(P_ema);
T = length(delta);
delta_m = zeros(T,1);
for i = window+1:T
    sub_wid = i-window:i;
    delta_m(i) = (delta(i) - mean(delta(sub_wid)))/std(delta(sub_wid));
end
stop_sel = zeros(size(delta));
stop_sel(delta_m>1) =stop_value(1);
stop_sel(delta_m<=1) =stop_value(2);


T = length(P);
RVol = zeros(T,1);
for i = window:T
    sub_wid = i-window+1:i;
    RVol(i) = std(P(sub_wid)-P_ema(sub_wid))*sqrt(window);
end

RVol_ema = EMA(RVol,window2);
ind = zeros(T,1);

for i = window+1:T-1
    if RVol(i)>RVol_ema(i)
        if P(i)>P_ema(i)
            ind(i) = 1;
            %continue
        elseif P(i)<P_ema(i)
            ind(i) = -1;
            %continue
        else
            ind(i) = 0;
            %continue
        end
    else
        ind(i) = 0;
        %continue
    end
    %ind(i+1) = ind(i);    
end

V = bac_test_CTA_update4(ind,[P_o,P],multi_val,ones(size(ind)),100000,4/10000,stop_sel);

figure;
bpcure_plot_update(tref,V);
title(infos);

[v,v_str,sta_val] = curve_static(V);
