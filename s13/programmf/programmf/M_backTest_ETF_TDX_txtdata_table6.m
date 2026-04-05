
%ETF回测 通达信txt
clear
close all

[~,~,info] = xlsread('data.xlsx','sheet3');
fns = cell(size(info,1),1);
for i = 1:length(info)
    fns{i} = sprintf('%s%s%s.txt',info{i,2},info{i,3},info{i,1}(1:6));
end

T = length(fns);
sta_re = [];
for index_sel = 1:T
    sprintf('%d-%d',index_sel,T)
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
    for i = 1:length(obj)
        obj(i) = plot(tref,V(:,i),'LineWidth',2,'color',colors(i,:));
        if eq(i,1)
            hold on
        end
    end
    setpixelposition(gcf,[416,397,961,420]);

    set(gca,'XTickLabelRotation',90);
    set(gca,'XTick',tref(floor(linspace(1,length(tref),40))),'xlim',tref([1,end]));
    datetick('x','yyyymmdd','keepticks');
    set(gca,'fontsize',12);

    box off
    set(gca,'linewidth',1.5);
    legend(obj,leg_str,'Location','northwest',...
        'NumColumns',length(obj))
    legend('boxoff')

    val = get(gca,'YTick');
    labels = num2str(val'*100,'%5.1f%%');
    set(gca,'yticklabel',labels)
    title(sub_index_name)

    Y = V;
    %Y = bsxfun(@minus,V,V(:,end));
    sub_re = zeros(1,size(Y,2)-1);
    for i = 1:size(Y,2)-1
        temp = cal_para_math(Y(end,[end,i]),tref(end)-tref(1)+1);
        sub_re(i) = temp(1)*100;
    end

    temp = cal_para_geo(cumsum(g_jump),tref(end)-tref(1)+1);
    sub_re = [{fns{index_sel}(end-9:end-4),fns{index_sel}(1:end-10)},num2cell([(tref(end)-tref(1)+1)/365,temp([1,3]),sub_re])];
    sta_re = cat(1,sta_re,sub_re);
end

sta_re

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