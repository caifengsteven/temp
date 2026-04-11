%M_com_Hindenburg
%整合中证2011年9月30日前数据再次计算
%版本2
%时间范围扩大
%用于恒生指数回测，数据基于excel
%历史数据已经按照格式排列好，不需要检索数据，计算很快。

%回测程序

clear
close all
%参数
window_cal = 20;
window_week = 5;
%载入数据
%index
[~,~,x_index] = xlsread('HSI_alldata.xlsx','index'); %指数数据
x_index = x_index(2:end,[1,end]);
%com
[~,~,x_com] = xlsread('HSI_alldata.xlsx','com'); %成份股名称数据
x_com = x_com(2:end,[1,2,4]);
x_com_t = datenum(x_com(:,1));

%data
[~,~,x] = xlsread('HSI_alldata.xlsx','data'); %成份股日内数据
symbol = x(2,2:end);
tref_str2 = x(3:end,1);
[tref_str,ia,ib] = intersect(x_index(:,1),tref_str2,'stable');
x_index = x_index(ia,:);
x = cell2mat(x(3:end,2:end))';
x = x(:,ib);
x_return = [zeros(size(x(:,1))),x(:,2:end)./x(:,1:end-1)-1];
tref_str2 = tref_str2(ib);
tref = datenum(tref_str);

X = nan(size(x)); %收益率数据
m = size(x,1);
T = length(tref_str);
for i = 1:m
    ia = strcmp(x_com(:,2),symbol(i));
    sub_x = x_com(ia,:);
    sub_x_t = x_com_t(ia,:);
    [sub_x_t,ia] = sort(sub_x_t);
    sub_x = sub_x(ia,:);
    [~,sub_x_t_order] = intersect(tref,sub_x_t,'stable');
    if length(sub_x_t_order)<length(sub_x_t)
        sub_x_t_order = cat(1,0,sub_x_t_order);
    end    
    sub_x_t_order = cat(1,sub_x_t_order,length(tref));
    for j = 2:length(sub_x_t_order)
        if strcmp(sub_x(j-1,end),'纳入')
            %股票上一时刻没有被剔除，可以记录收益
            sub_ind = sub_x_t_order(j-1)+1:sub_x_t_order(j);
            X(i,sub_ind) = x_return(i,sub_ind);
        end
    end
end

close_price = cell2mat(x_index(:,end));
%y0 = cell2mat(index_data(:,end));
y0 = [0;close_price(2:end)./close_price(1:end-1)-1];
%%{
%}
%%{
%合成趋同度因子
Y_pre = nan(size(X));
for i = 1:m
    sub_x = X(i,:);
    for j = window_cal:T
        sub_sub_x = sub_x(j-window_cal+1:j);
        sub_sub_y = y0(j-window_cal+1:j);
        window_ind_sub = ~isnan(sub_sub_x);
        sub_sub_x = sub_sub_x(window_ind_sub);
        sub_sub_y = sub_sub_y(window_ind_sub);  
        if length(sub_sub_x)>5
            Y_pre(i,j) = get_rsqure(sub_sub_x',sub_sub_y);
        end
    end
    sprintf('%d-%d',i,m)
end
%}

%cal indicator
factor_v = nan(T,1);
for i = 1:T
    sub_y = Y_pre(:,i);
    sub_y(isnan(sub_y)) = [];
    if ~isempty(sub_y)
        factor_v(i) = mean(sub_y);
    end
end

% re = [];
% re.X = X;
% re.Y_pre = Y_pre;
% re.tref = tref;
% re.symbol_all = symbol;
% re.close_price = close_price;
% re.factor_v = factor_v;

t_ind = 1:length(tref);

figure;
y_lim = nan(size(y0));
x_lim = y_lim;
y_lim(window_week+1:end) = close_price(window_week+1:end)./close_price(1:end-window_week)-1;
x_lim(window_week+1:end) = factor_v(window_week+1:end)./factor_v(1:end-window_week)-1;
%计算坐标轴象限位置
v_location = zeros(size(y_lim));
v_location(x_lim>0&y_lim>0) = 1;
v_location(x_lim<0&y_lim>0) = 2;
v_location(x_lim<0&y_lim<0) = 3;
v_location(x_lim>0&y_lim<0) = 4;
%计算转移矩阵
%v_location = v_location(t_ind);
v_p = zeros(4,4);
for i = 1:4
    sub_ind = find(eq(v_location,i));
    if eq(sub_ind(end),length(v_location))
        sub_ind(end) = [];
    end
    for j = 1:4
        v_p(i,j) = sum(eq(v_location(sub_ind+1),j))/length(sub_ind)*100;        
    end
end
%相邻干扰，和文献结果不同，对回测无影响，有时间可以找文献做下

plot(x_lim(t_ind)*100,y_lim(t_ind)*100,'r.')
hold on
lims = axis(gca);
lims1 = [min(lims),max(lims)];
plot(lims1,[0,0],'-','LineWidth',2,'color',[0.47,0.67,0.19])
plot([0,0],lims1,'-','LineWidth',2,'color',[0.47,0.67,0.19]);
axis(lims);
xlabel('趋同度涨跌幅')
ylabel('指数涨跌幅')

figure
t_ind1 = 1:length(tref);
yyaxis left
plot(tref(t_ind1),close_price(t_ind1),'-','linewidth',2,'color',[0.64 0.08 0.18]);
yyaxis right
plot(tref(t_ind1),factor_v(t_ind1),'-','LineWidth',2,'color',[0,0.451,0.7412]);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(t_ind1(floor(linspace(1,length(t_ind),30)))),'xlim',tref(t_ind1([1,end])));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);
box off
set(gca,'linewidth',1.5);

legend({'指数','趋同度指数'})

T = length(tref);
bac_re_return = zeros(T,1);
signal = zeros(T,1);
for i = window_week+1:T
    %信号位置
    sub_pos = i- window_week;
    %判断当前信号
    if any(eq([1,3],v_location(sub_pos)))
        signal(i) = 1;
    elseif any(eq([2,4],v_location(sub_pos)))
        signal(i) = -1;
    else
        signal(i) = 0;
    end
    
    bac_re_return(i) = y_lim(i)*signal(i);
end
bac_re = cumprod(1+bac_re_return/5);

figure

plot(tref(t_ind),close_price(t_ind)/close_price(t_ind(1)),'-','linewidth',2,'color',[0,0.451,0.7412]);
hold on
plot(tref(t_ind),bac_re(t_ind),'-','LineWidth',2,'color',[0.64 0.08 0.18]);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(t_ind(floor(linspace(1,length(t_ind),30)))),'xlim',tref(t_ind([1,end])));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);
box off
set(gca,'linewidth',1.5);

legend({'指数净值','策略净值'})
set(gca,'YGrid','on')
[v,v_str,sta_val] = curve_static(bac_re(t_ind));