%期货合约
%使用几何收益率计算
%增加合约换月处理
%使用对数收益
%回测
%update 20200402 数据升级  收盘做多、开盘平仓

clear
%股指数据
key_str = '股指期货高开T0验证';
index_name_pool = {'沪深300股指期货','上证50股指期货','中证500股指期货'};
index_code = {'IF','IH','IC'};    
index_pool = {'CFFEX','CFFEX','CFFEX'};  
T_index = length(index_name_pool);
h = figure;
h.Name = key_str;
for index_sel = 1:T_index
  
sub_index_name = index_name_pool{index_sel};
t0 = cell(size(index_name_pool));
t0{1} = '2014-05-01';

sql_str_f1 = ['select tradeDate,ticker,openprice,closeprice from yuqerdata.yq_MktMFutdGet  ',...
    'where contractObject = ''%s''  and mainCon=1 and tradeDate>=''%s'' order by tradedate'];

sql_str_f2 = ['select tradeDate,ticker,openprice,closeprice from yuqerdata.yq_MktMFutdGet  ',...
    'where contractObject = ''%s''  and mainCon=1 order by tradedate'];

if isempty(t0{index_sel})
    x = fetchmysql(sprintf(sql_str_f2,index_code{index_sel}),2);
else
    x = fetchmysql(sprintf(sql_str_f1,index_code{index_sel},t0{index_sel}),2);
end



index_contracts_num = cellfun(@(x) str2double(x(length(index_code{index_sel})+1:end)),x(:,2));
index_contracts_num = [0;diff(index_contracts_num)];
index_contracts_num = ~eq(index_contracts_num,0);
index_data= x(:,[1,3,4]);


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

g_cum(index_contracts_num) = 0;
g_jump(index_contracts_num) = 0;
g_inner(index_contracts_num) = 0;


g_data_geo = [g_cum,g_inner,g_jump];
%算术收益率
g_cum_m = [0;close_price(2:end)./close_price(1:end-1)-1];
g_jump_m = [0;open_price(2:end)./close_price(1:end-1)-1];
g_inner_m = close_price./open_price-1;
g_inner_m(1) = 0;

g_cum_m(index_contracts_num) = 0;
g_jump_m(index_contracts_num) = 0;
g_inner_m(index_contracts_num) = 0;

g_data_math = [g_cum_m,g_inner_m,g_jump_m];


fee = [1.5/10000,3/10000];
g1 = g_jump;
g1(~index_contracts_num) = g1(~index_contracts_num)-fee(1)*2;
g2 = g_jump;
g2(~index_contracts_num) = g2(~index_contracts_num)-fee(2)*2;


%obj = zeros(3,1);
colors = [0.64,0.78,0.18;0.93,0.69,0.13;ones(1,3)*0.65];
g_info = {'无手续费','手续费万一点五','手续费万三','基准'};
subplot(T_index,1,index_sel)
obj = plot(tref,1+cumsum([g_jump,g1,g2,g_cum]),'-','linewidth',2);
obj(1).Color = 'r';
obj(end).Color = colors(3,:);
%obj(1)= plot(tref,cumprod(1+g_jump_m),'r-','linewidth',2);
%hold on
%obj(2)= plot(tref,cumprod(1+g_cum_m),'k-','linewidth',2,'color',colors(3,:));
%obj(1)= plot(tref,cumsum([g_cum,g_inner,g_jump]),'linewidth',2);
%plot(tref([1,end]),[0,0],'k-','LineWidth',2);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(floor(linspace(1,length(tref),40))),'xlim',tref([1,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);

box off
set(gca,'linewidth',1.5);
legend(obj,g_info,'Location','northwest',...
    'NumColumns',length(obj),'location','northwest')
legend('boxoff')

val = get(gca,'YTick');
labels = num2str(val'*100,'%6.1f%%');
% Adjust labels on plot
%set(gca,'yticklabel',labels)
title(sub_index_name)
end

%setpixelposition(gcf,[416,397,961,420]);



