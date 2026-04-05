%{
我们给出上
市超过 5 年、且过去 5 年每日平均成交额超过 300 万元的 ETF 使用半仓 T0 策略的超额
收益情况。
%}
%ETF回测
%对接优矿数据
clear

keystr = 'ETFT0半仓策略验证';
[~,~,info] = xlsread('data.xlsx','sheet3');
index_name_pool = cellfun(@(x,y) [x,y],info(:,2),info(:,3),'UniformOutput',false);
index_code_pool = cellfun(@(x) x(1:6),info(:,1),'UniformOutput',false);


%index_name_pool = {'华夏上证50ETF';'嘉实沪深300ETF';'华泰柏瑞沪深300ETF';'南方中证500ETF';'易方达创业板 ETF'};
%index_code_pool = {'510050','159919','510300','510500','159915'};
%t_0 = {[],'2012-05-28',[],'2013-03-15',[]};
T_index = length(index_code_pool);

sql_str_f1 = ['select tradedate,openprice*accumAdjFactor,closeprice*accumAdjFactor from yuqerdata.MktFunddGet ',...
    'where ticker = ''%s'' and tradeDate>=''%s'' order by tradeDate'];

for index_sel = 1:T_index
    sub_index_name = index_name_pool{index_sel};
    sub_index_code = index_code_pool{index_sel};
    %sub_t0 = t_0{index_sel};
    %if isempty(sub_t0)
    sub_t0 = '2000-01-01';
    %end
    index_data = fetchmysql(sprintf(sql_str_f1,sub_index_code,sub_t0),2);

    tref_str = index_data(:,1);
    tref = datenum(tref_str);
    o_c_price = cell2mat(index_data(:,2:3));
    open_price = o_c_price(:,1);
    close_price = o_c_price(:,2);
    %g_cum; g_jump g_inner %累计收益，跳价收益，日内收益
    %几何收益率
    g_cum = [0;log(close_price(2:end)./close_price(1:end-1))];
    g_jump = [0;log(open_price(2:end)./close_price(1:end-1))];
    g_jump(isinf(g_jump)) = 0;
    
    g_inner = log(close_price./open_price);
    g_inner(1) = 0;
    g_inner(isinf(g_inner)) = 0;
    
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
    g_jump_new(isinf(g_jump_new)) = 0;

    leg_str = {'无手续费','手续费万三','手续费万五','手续费千一','基准'};
    fee_all = [0,3,5,10]./10000;
    V = zeros(length(tref),length(fee_all)+1);
    V(:,end) = close_price./close_price(1);
    for i = 1:length(fee_all)
        V(:,i) = get_half_r(g_jump_new,fee_all(i));
    end
    colors = [0.6392,0.0784,0.1804;0.93,0.69,0.13;ones(1,3)*0.65;ones(1,3)*0.5; 0.3020,0.7490,0.9294];
    obj = zeros(5,1);
    %subplot(ceil(T_index/2),2,index_sel)
    index_mod = mod(index_sel,6);
    if eq(index_mod,1)
        h=figure;
        h.Name = keystr;
    elseif eq(index_mod,0)
        index_mod = 6;
    end
    subplot(3,2,index_mod)
    for i = 1:length(obj)
        obj(i) = plot(tref,V(:,i),'LineWidth',2,'color',colors(i,:));
        if eq(i,1)
            hold on
        end
    end
    setpixelposition(gcf,[416,397,961,420]);

    set(gca,'XTickLabelRotation',90);
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
    a = bsxfun(@minus,V,V(:,end));
    a = a(:,1:end-1);
    a = a(end,:) -a(1,:);
    %a = V(end,:) -V(end,end);
    a./(tref(end)-tref(1)+1)*365*100;

    N = tref(end)-tref(1)+1;
    b = real(a.^(365/N)-1)*100;
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