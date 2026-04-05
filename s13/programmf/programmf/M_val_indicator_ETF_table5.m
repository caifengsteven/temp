%ETF数据
%数据来源 通达信
%使用几何收益率计算
clear
close all
index_name_pool = {'华夏上证50ETF510050.csv';'华泰柏瑞沪深300ETF510300.csv';...
    '南方中证500ETF510500.csv';'嘉实沪深300ETF159919.csv';'易方达创业板 ETF159915.csv'};

sta_re = [];
for index_sel = 1:length(index_name_pool)
    sub_index_name = index_name_pool{index_sel}(1:end-4);
    %指数数据
    % t0 = '2005-01-01';
    % tt = '2019-03-29';
    % index_data1 = get_index_data_ycz(sub_index_name,t0);
    % index_data2 = get_index_data_JJ(sub_index_name,index_data1{end,1},tt);
    % index_data = [index_data1;index_data2(2:end,:)];
    [~,~,index_data] = xlsread(sprintf('%s.csv',sub_index_name));
    index_data = index_data(5:end-1,[1,2,5]);
    tref = datenum(index_data(:,1));
    ia = tref<=datenum(2019,3,29);
    tref = tref(ia,:);
    index_data = index_data(ia,:);


    tref_str = index_data(:,1);
    %tref = datenum(tref_str);
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

    V = V';
    V = V([3,2,1],:);
    V = V(:,[1,3,2]);
    V = num2cell(V(:,[1,1:end]));
    V(:,1) = {sub_index_name};
    sta_re = cat(1,sta_re,V);
end
