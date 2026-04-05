%指数回测验证
%对接优矿数据
%指数半仓策略  
%{
指数半仓 T0 策略测试方法： 1、初始净值假设为 1，定义基准为指数本身，交易时扣除双边手续费，即买入卖出均收
取手续费； 2、在第一日开盘时买入半仓股票，第二天开盘时再补至满仓，收盘时卖出一半股票，恢
复半仓状态，如此循环往复；
3、计算不同手续费率下策略净值走势和绩效表现； 4、年化超额收益定义为策略与基准的年化收益之差；
%}
clear
key_str = '指数低开T0策略验证';

index_name_pool = {'沪深300','上证指数','上证50','中证500','深证成指',...
    '创业板指','中小板指','中证1000',...
        '深次新股','中证流通'};
T_index = length(index_name_pool);
for index_sel = 1:T_index    
    sub_index_name = index_name_pool{index_sel};
    %指数数据
    t0 = '2005-01-01';
    tref = yq_methods.get_tradingdate(t0,datestr(now,'yyyy-mm-dd'));
    tt = tref{end};

    [index_data,index_code] = get_index_data_yuqer(sub_index_name,t0);
    if ~isempty(index_data)
        temp_t = index_data{end,1};
    else
        temp_t = t0;
    end

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
    g_jump_new(isnan(g_jump_new)) = 0;

    leg_str = {'无手续费','手续费万三','手续费万五','基准'};
    fee_all = [0,3,5]./10000;
    V = zeros(length(tref),4);
    V(:,end) = close_price./close_price(1);
    for i = 1:length(fee_all)
        V(:,i) = get_half_r(g_jump_new,fee_all(i));
    end
    index_mod = mod(index_sel,6);
    if eq(index_mod,1)
        h=figure;
        h.Name = key_str;
    elseif eq(index_mod,0)
        index_mod = 6;
    end
    subplot(3,2,index_mod)
    
    colors = [0.6392,0.0784,0.1804;0.93,0.69,0.13;ones(1,3)*0.65;ones(1,3)*0.5];
    obj = zeros(4,1);
    for i = 1:length(obj)
        obj(i) = plot(tref,V(:,i),'LineWidth',2,'color',colors(i,:));
        if eq(i,1)
            hold on
        end
    end

    set(gca,'XTickLabelRotation',45);
    set(gca,'XTick',tref(floor(linspace(1,length(tref),20))),'xlim',tref([1,end]));
    datetick('x','yyyymmdd','keepticks');
    set(gca,'fontsize',12);

    box off
    set(gca,'linewidth',1.5);
    legend(obj,leg_str,'Location','northwest',...
        'NumColumns',length(obj),'location','best')
    legend('boxoff')

    val = get(gca,'YTick');
    labels = num2str(val'*100,'%5.1f%%');
    %set(gca,'yticklabel',labels)
    title(sub_index_name)
    Y = V;
    %Y = bsxfun(@minus,V,V(:,end));
    sub_re = zeros(1,size(Y,2)-1);
    for i = 1:size(Y,2)-1
        temp = cal_para_math(Y(end,[4,i]),tref(end)-tref(1)+1);
        sub_re(i) = temp(1);
    end
end
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

end

function[v,v_str] = cal_para_math(y,N)
% y = cumprod(1+rand(1000,1)/1000);
%(AC3277/100)^(244/COUNT(AC120:AC3277))-1
%1年化收益率
v_str{1} = '年化收益率';
v(1) = (y(end)/y(1))^(365/N)-1;
end