%{
测试个股使用半仓 T0 策略的收益情况。测试过程中剔除了交易
年数不满 5 年的股票，并将最早测试日期统一定为 2005 年 1 月，这就意味着在 2005 年 1
月前已经上市的股票从这个时点开始测试，在 2005 年 1 月之后上市的股票取全部历史数
据进行测试。由于股票交易需要缴纳印花税，相比 ETF 交易成本更高，我们重点观测在
双边千一手续费下的测试结果，这个手续费相对比较符合现实情况。
%}
clear
key_str = 'S13股票低开验证';
[~,~,info] = xlsread('data.xlsx','sheet1');
symbol = info(:,1);
%shortname = info(:,2);
T_symbol = length(symbol);
sta_re = cell(size(symbol));
sql_str_f1 = ['select tradeDate,openprice,closeprice from yuqerdata.yq_dayprice',...
    ' where symbol=''%s'' order by tradeDate'];
sql_str_f2 = ['select tradeDate,accumAdjFactor from yuqerdata.MktEqudAdjAfGet ',...
    'where ticker = ''%s'' order by tradeDate'];
sql_str_f3 = 'select secShortName from yuqerdata.EquGet where ticker = ''%s'' limit 1';
title_str = {'代码','简称','测试年数','年化跳价(%)','策略年化(%)','基准年化(%)','年化超额收益(%)'};

parfor astock_sel = 1:length(symbol)
   
    sub_info = symbol{astock_sel};
    sub_index_name = sprintf('S%sSE.%s',sub_info(end),sub_info(1:6));
    sub_info=strsplit(sub_info,'.');
    sub_info = sub_info{1};
    shortname = fetchmysql(sprintf(sql_str_f3,sub_info),2);
    
    sub_x1 = fetchmysql(sprintf(sql_str_f1,sub_info),2);
    sub_x2 = fetchmysql(sprintf(sql_str_f2,sub_info),2);
    [~,ia,ib] = intersect(sub_x1(:,1),sub_x2(:,1));
    
    sub_x1 = sub_x1(ia,:);
    sub_x2 = sub_x2(ib,:);
    sub_x3 = bsxfun(@times,cell2mat(sub_x1(:,2:end)),cell2mat(sub_x2(:,end)));
    index_data = [sub_x1(:,1),num2cell(sub_x3)];

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
%     colors = [0.6392,0.0784,0.1804;0.93,0.69,0.13;ones(1,3)*0.65;ones(1,3)*0.5; 0.3020,0.7490,0.9294];
%     obj = zeros(5,1);
%     figure;
%     for i = 1:length(obj)
%         obj(i) = plot(V(:,i),'LineWidth',2,'color',colors(i,:));
%         if eq(i,1)
%             hold on
%         end
%     end
%     setpixelposition(gcf,[416,397,961,420]);
% 
%     set(gca,'XTickLabelRotation',90);
%     set(gca,'XTick',floor(linspace(1,length(tref),40)),'xlim',[1,length(tref)]);
%     set(gca,'XTickLabel',cellstr(datestr(tref(floor(linspace(1,length(tref),40))),'yyyymmdd')));
%     %datetick('x','yyyymmdd','keeplimits');
%     set(gca,'fontsize',12);
% 
%     box off
%     set(gca,'linewidth',1.5);
%     legend(obj,leg_str,'Location','northwest',...
%         'NumColumns',length(obj),'location','best')
%     legend('boxoff')
% 
%     % val = get(gca,'YTick');
%     % labels = num2str(val'*100,'%5.1f%%');
%     % set(gca,'yticklabel',labels)
%     title(sub_index_name)

    Y = V;

    %年化跳价
    temp1 = cal_para_geo(cumsum(g_jump),tref(end)-tref(1)+1);
    %策略年化
    temp2 = cal_para_math(Y(:,4),tref(end)-tref(1)+1)*100;
    %基准年化
    temp3 = cal_para_math(Y(:,5),tref(end)-tref(1)+1)*100;
    sub_re = [(tref(end)-tref(1)+1)/365,temp1(1),temp2(1),temp3(1),temp2(1)-temp3(1)];
    sub_re = [{sub_index_name,shortname{1}},num2cell(sub_re)];
    
    sta_re{astock_sel} = sub_re';
    sprintf('%s:%d-%d',key_str,astock_sel,T_symbol)
end
%sub_re = [{fns{index_sel}(end-9:end-4),fns{index_sel}(1:end-10)},num2cell([(tref(end)-tref(1)+1)/365,temp([1,3]),sub_re])];
%sta_re = cat(1,sta_re,sub_re);
sta_re = [sta_re{:}]';

gui_result(sta_re,'S13A股低开半仓策略收益统计',title_str)

sta_re = [title_str;sta_re];
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