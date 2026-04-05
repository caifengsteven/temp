clear
close all

sub_index_name = 'SZSE.300033';
%sub_index_name = 'SZSE.300377';
%sub_index_name = 'SZSE.300226';
%sub_index_name = 'SHSE.600892';
%后复权
sql_str = 'select tradingdate,openprice,closeprice from futuredata.jjastockdata_adjust_prev where symbol = ''%s'' and tradingdate>=''2005-01-01'' order by tradingdate';
%不复权
%sql_str = 'select tradingdate,openprice,closeprice from futuredata.jjastockdata_adjust_none where symbol = ''%s'' and tradingdate>=''2005-01-01'' and tradingdate<=''2019-03-29'' order by tradingdate';
index_data = fetchmysql(sprintf(sql_str,sub_index_name),2);

% sub_index_name = '300226';
% sql_str = 'select tradingdate,openprice,closeprice from futuredata.STK_MKT_BWARDQUOTATION where symbol = ''%s'' and tradingdate>=''2005-01-01''  and filling = 0 order by tradingdate';
% index_data = fetchmysql(sprintf(sql_str,sub_index_name),2);

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
%
g_jump_new = [0;close_price(2:end)./open_price(1:end-1)-1];

leg_str = {'无手续费','手续费万三','手续费万六','手续费千一','基准'};
fee_all = [0,3,6,10]./10000;
V = zeros(length(tref),length(fee_all)+1);
V(:,end) = close_price./close_price(1);
for i = 1:length(fee_all)
    V(:,i) = get_half_r(g_jump_new,fee_all(i));
end
colors = [0.6392,0.0784,0.1804;0.93,0.69,0.13;ones(1,3)*0.65;ones(1,3)*0.5; 0.3020,0.7490,0.9294];
obj = zeros(5,1);
figure;
for i = 1:length(obj)
    obj(i) = plot(V(:,i),'LineWidth',2,'color',colors(i,:));
    if eq(i,1)
        hold on
    end
end
setpixelposition(gcf,[416,397,961,420]);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',floor(linspace(1,length(tref),40)),'xlim',[1,length(tref)]);
set(gca,'XTickLabel',cellstr(datestr(tref(floor(linspace(1,length(tref),40))),'yyyymmdd')));
%datetick('x','yyyymmdd','keeplimits');
set(gca,'fontsize',12);

box off
set(gca,'linewidth',1.5);
legend(obj,leg_str,'Location','northwest',...
    'NumColumns',length(obj),'location','best')
legend('boxoff')

% val = get(gca,'YTick');
% labels = num2str(val'*100,'%5.1f%%');
% set(gca,'yticklabel',labels)
title(sub_index_name)

Y = V;

%年化跳价
temp1 = cal_para_geo(cumsum(g_jump),tref(end)-tref(1)+1);
%策略年化
temp2 = cal_para_math(Y(:,4),tref(end)-tref(1)+1);
%基准年化
temp3 = cal_para_math(Y(:,5),tref(end)-tref(1)+1);
sub_re = [(tref(end)-tref(1)+1)/365,temp1(1),temp2(1)*100,temp3(1)*100]


%sub_re = [{fns{index_sel}(end-9:end-4),fns{index_sel}(1:end-10)},num2cell([(tref(end)-tref(1)+1)/365,temp([1,3]),sub_re])];
%sta_re = cat(1,sta_re,sub_re);

function c_new = get_half_r(g_jump_new,fee)
if nargin < 2
    fee = 0;
end
g_r_1 = zeros(size(g_jump_new)); %相当于间隔一天的两个半仓策略
g_r_2 = g_r_1;
g_r_1(2:2:end) = g_jump_new(2:2:end);
g_r_1 = g_r_1 - fee;
g_r_2(3:2:end) = g_jump_new(3:2:end);
g_r_2(2:end) = g_r_2(2:end)-fee;

c_new = cumprod(1+g_r_1)*0.5+cumprod(1+g_r_2)*0.5;
%c_new = cumprod(1+g_r_1*0.5)+cumprod(1+g_r_2*0.5);
end