%扩展至所有期货
%扩展股指数据
%扩展国债数据
clear
close all

t0 = '2013-01-04';
tt = '2019-12-30';

code_all = {'DCE','SHFE','CZCE','CFFEX'};

type1_sel = 4;%选择商品交易所 1 大商所 2上商所 3郑商所 4 股指或者国债期货
infos = '10年期国债';%选择交易类别 类别信息可以见后文的import cm data fun中的定义

[f_name,M_V] = import_cm_data();

type1 = code_all{type1_sel};
type2 = f_name{type1_sel}(infos);

db_name = 'futuredata';
window =25;
window2 =5;

stop_value = [2.5,5]/100;

multi_val = M_V{type1_sel}(infos);

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

if type1_sel<4
    cash0 = 100000;
else
    cash0 = 1000000;
end


V = bac_test_CTA_update4(ind,[P_o,P],multi_val,ones(size(ind)),cash0,4/10000,stop_sel);

figure;
bpcure_plot_update(tref,V);
title(infos);

[v,v_str,sta_val] = curve_static(V);



function [f_name,M_V] = import_cm_data()
%大商所
f_name{1}= containers.Map({'豆一';'豆二';'豆粕';'豆油';'棕榈油';'玉米';'玉米淀粉';'鸡蛋';'纤维板';'胶合板';'聚乙烯';'聚氯乙烯';'聚丙烯';'焦炭';'焦煤';'铁矿石';'乙二醇'},...
    {'A';'B';'M';'Y';'P';'C';'CS';'JD';'FB';'BB';'L';'V';'PP';'J';'JM';'I';'EG'});
M_V{1} = containers.Map({'豆一';'豆二';'豆粕';'豆油';'棕榈油';'玉米';'玉米淀粉';'鸡蛋';'纤维板';'胶合板';'聚乙烯';'聚氯乙烯';'聚丙烯';'焦炭';'焦煤';'铁矿石';'乙二醇'},...
    [10,10,10,10,10,10,10,5,500,500,5,5,5,100,60,100,10]);
%上商所对应名称
f_name{2}= containers.Map({'铜';'铝';'锌';'铅';'镍';'锡';'黄金';'白银';'螺纹钢';'线材';'热轧卷板';'原油';'燃料油';'沥青';'天然橡胶';'纸浆'},...
    {'CU';'AL';'ZN';'PB';'NI';'SN';'AU';'AG';'RB';'WR';'HC';'SC';'FU';'BU';'RU';'SP'});
M_V{2} = containers.Map({'铜';'铝';'锌';'铅';'镍';'锡';'黄金';'白银';'螺纹钢';'线材';'热轧卷板';'原油';'燃料油';'沥青';'天然橡胶';'纸浆'},...
    [5,5,5,5,1,1,1000,15,10,10,10,1000,50,10,10,10]);
%郑商所对应名称
f_name{3} = containers.Map({'棉花','早籼','菜油','白糖','强麦','硬麦','菜油','早籼','强麦','玻璃','普麦','油菜籽','菜籽粕',...
    '动力煤','粳稻','甲醇','晚籼','硅铁','锰硅','棉纱','苹果','PTA'},...
    {'CF','ER','RO','SR','WS','WT','OI','RI','WH','FG','PM','RS','RM',...
    'TC','JR','MA','LR','SF','SM','CY','AP','TA'});
M_V{3} = containers.Map({'棉花','早籼','菜油','白糖','强麦','硬麦','菜油','早籼','强麦','玻璃','普麦','油菜籽','菜籽粕',...
    '动力煤','粳稻','甲醇','晚籼','硅铁','锰硅','棉纱','苹果','PTA'},...
    [5,20,10,10,20,20,10,20,20,20,50,10,10,100,20,10,20,5,5,5,10,5]);
f_name{4} = containers.Map({'沪深300股指期货','中正500股指期货','上证50股指期货','5年期国债','10年期国债'},...
    {'IF','IC','IH','TF','T'});
M_V{4} = containers.Map({'沪深300股指期货','中正500股指期货','上证50股指期货','5年期国债','10年期国债'},...
    [300,200,300,10000,10000]);
end