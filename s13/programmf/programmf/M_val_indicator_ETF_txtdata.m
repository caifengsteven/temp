%{
1 交银180治理ETF510010.txt
2 华夏上证主要消费ETF510630.txt
3 华夏上证医药卫生ETF510660.txt
4 华夏上证金融地产ETF510650.txt
5 华夏中小板ETF159902.txt
6 华夏沪深300ETF510330.txt
7 华安上证180ETF510050.txt
8 华安上证龙头ETF510190.txt
9 华宝上证180价值ETF510030.txt
10 华泰柏瑞红利ETF510880.txt
11 南方上证380ETF510290.txt
12 南方小康产业ETF510160.txt
13 南方开元沪深300ETF159925.txt
14 南方深成ETF159903.txt
15 博时自然资源ETF510410.txt
16 博时超大盘ETF510020.txt
17 嘉实中证500ETF159922.txt
18 国泰上证180金融ETF510230.txt
19 国联安上证商品ETF510170.txt
20 工银上证央企50ETF510060.txt
21 工银瑞信深证红利ETF159905.txt
22 广发中小板300ETF159907.txt
23 广发中证500ETF510510.txt
24 易方达上证中盘ETF510130.txt
25 易方达沪深300ETF510310.txt
26 易方达沪深300医药卫生ETF512010.txt
27 海富通上证周期ETF510110.txt
28 诺安上证新兴产业ETF510260.txt
29 鹏华深证民营ETF159911.txt
%}

%ETF数据
%数据来源 通达信
%数据格式txt
clear
close all

fns = dir(fullfile('data','*.txt'));
fns = {fns.name}';
T = length(fns);
index_sel = 2;
if index_sel>T
    sprintf('输入文件序号大于文件数目，最大为%d',T)
    return
end

sub_fn = fullfile('data',fns{index_sel});

[index_data,sub_index_name] = import_tdx_txtdata(sub_fn);

tref_str = index_data(:,1);
tref = datenum(tref_str);
o_c_price = cell2mat(index_data(:,2:3));
open_price = o_c_price(:,1);
close_price = o_c_price(:,2);
%g_cum; g_jump g_inner %累计收益，跳价收益，日内收益
%几何收益率
g_cum = [0;log(close_price(2:end)./close_price(1:end-1))];
g_jump = [0;log(open_price(2:end)./close_price(1:end-1))];
g_inner = log(close_price./open_price);
g_inner(1) = 0;

g_info = {'累计对数收益','累计日内收益','累计跳价收益'};
g_data_geo = [g_cum,g_inner,g_jump];
%算术收益率
g_cum_m = [0;close_price(2:end)./close_price(1:end-1)-1];
g_jump_m = [0;open_price(2:end)./close_price(1:end-1)-1];
g_inner_m = close_price./open_price-1;
g_inner_m(1) = 0;
g_data_math = [g_cum_m,g_inner_m,g_jump_m];

obj = zeros(3,1);
colors = [0.64,0.78,0.18;0.93,0.69,0.13;ones(1,3)*0.65];
obj(1)= plot(tref,cumsum(g_cum),'-','linewidth',2,'color','r');
hold on
obj(2)= plot(tref,cumsum(g_inner),'-','linewidth',2,'color',colors(2,:));
obj(3)= plot(tref,cumsum(g_jump),'-','linewidth',2,'color',colors(3,:));
%obj(1)= plot(tref,cumsum([g_cum,g_inner,g_jump]),'linewidth',2);
plot(tref([1,end]),[0,0],'k-','LineWidth',2);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(floor(linspace(1,length(tref),40))),'xlim',tref([1,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);

box off
set(gca,'linewidth',1.5);
legend(obj,g_info,'Location','northwest',...
    'NumColumns',length(obj),'location','best')
legend('boxoff')

val = get(gca,'YTick');
labels = num2str(val'*100,'%5.1f%%');
% Adjust labels on plot
set(gca,'yticklabel',labels)
title(sub_index_name)
setpixelposition(gcf,[416,397,961,420]);

%[v,v_str] = curve_static(-cumprod(1+g_jump_m))
V = zeros(3,3);
for i = 1:3
    %sprintf('%s',g_info{i})
    [V(:,i),v_str] = cal_para_geo(cumsum(g_data_geo(:,i)),tref(end)-tref(1)+1);
end

f_str = '%s结果\n\t\t\t 年化收益率\t\t Sharp值 \t\t年化波动率\n';
for i = 1:3
    f_str = [f_str,g_info{i},'\t%0.3f \t\t %0.3f \t\t %0.3f \n'];
end

sprintf(f_str,sub_index_name,V)

