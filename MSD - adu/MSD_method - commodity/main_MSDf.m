%{
MSD
知情投资者情绪(Informed Trader Sentiment, ITS)指标和非知情投资者情绪
(Uninformed Trader Sentiment, UTS)指标，并在此基础上计算市场情绪差异
(Market Sentiment Difference, MSD)指标
%}
clear
%载入数据
%X1 排名前20的结算会员
X1 =readtable('msd_data.xlsx');
%X2 总信息
X2 = readtable('msd_data_total.xlsx');
% 时间
tref_str = unique(X1.tradingdate);
tref = datenum(tref_str);
%变量名称
var_names1 = {'volume_name','volume_value','buy_name','buy_value','sail_name','sail_value','code'};
var_names2 = {'total_volume','buy_volume','code'};
T = length(tref_str);
S = cell(T,1);
L = zeros(T,1);
ITS = zeros(T,1);
UTS = zeros(T,1);
for i = 1:T
    sub_x = X1(eq(X1.tradingdate,tref_str(i)),var_names1);
    sub_x_t =X2(eq(X2.tradingdate,tref_str(i))&strcmp(X2.type,{'期货公司'}),var_names2);
    sub_code = sort(sub_x.code);
    sub_x = sub_x(strcmp(sub_x.code,sub_code(1)),:);
    sub_x_t = sub_x_t(strcmp(sub_x_t.code,sub_code(1)),1:2);
    
    sub_x = table2cell(sub_x);
    sub_x_t = table2array(sub_x_t);
        
    sub_x(:,1) = cellfun(@(x,y) [x,y],sub_x(:,7),sub_x(:,1),'UniformOutput',false);
    sub_x(:,3) = cellfun(@(x,y) [x,y],sub_x(:,7),sub_x(:,3),'UniformOutput',false);
    sub_x(:,5) = cellfun(@(x,y) [x,y],sub_x(:,7),sub_x(:,5),'UniformOutput',false);
    
    inds = suscc_intersect({sub_x(:,1),sub_x(:,3),sub_x(:,5)});
    sub_x1 = [];
    sub_x2 = [];
    for j = 1:size(inds,2)
        sub_inds1 = inds(:,j);
        sub_inds2 = 1:size(sub_x,1);
        sub_inds2(sub_inds1) = [];
        sub_x1 = cat(2,sub_x1,cell2mat(sub_x(sub_inds1,j*2)));
        sub_x2 = cat(2,sub_x2,cell2mat(sub_x(sub_inds2,j*2)));
    end
    %sub_x volume_value,buy_value,sail_value
    sub_x = [sub_x1;sum(sub_x2)];
    S_1 = (sub_x(:,2)+sub_x(:,3))./sub_x(:,1);
    S_2 = sub_x_t(2)/sub_x_t(1)*2;
    
    %sub_x_t2 = sum(sub_x);
    %S_2 = sub_x_t2(2)/sub_x_t2(1);
    if all(eq(S_1,0))
        ITS(i) = nan;
        UTS(i) = nan;
    else
        sub_ind1 = S_1>S_2;

        sub_x_ITS = sub_x(sub_ind1,:);
        sub_x_UTS = sub_x(~sub_ind1,:);
        %L(i) = size(sub_x_UTS,1);
        if ~isempty(sub_x_ITS)
            sub_x_ITS_s = sum(sub_x_ITS,1);
            ITS(i) = (sub_x_ITS_s(2)-sub_x_ITS_s(3))/(sub_x_ITS_s(2)+sub_x_ITS_s(3));
        end
        if ~isempty(sub_x_UTS)
            sub_x_UTS_s = sum(sub_x_UTS,1);
            UTS(i) = (sub_x_UTS_s(2)-sub_x_UTS_s(3))/(sub_x_UTS_s(2)+sub_x_UTS_s(3));
        end
    end
end
nan_ind = find(isnan(UTS));
nan_ind(eq(nan_ind,1)) = [];
UTS(nan_ind) = UTS(nan_ind-1);
nan_ind = find(isnan(ITS));
nan_ind(eq(nan_ind,1)) = [];
ITS(nan_ind) = ITS(nan_ind-1);
%}
%三种指标计算
mod_str = containers.Map([1,2,3],{'ITS','UTS','MSD'});
mod= mod_str(1);

%止损、止跌条件
stop_losses = -0.03;
takeprofit  = 0.10;
% stop_losses = -inf;
% takeprofit = inf;

if strcmp(mod,'ITS')
    r = 0;
    data = [tref,ITS];
    dir_sel = '+';
elseif strcmp(mod,'UTS')
    r = 0;
    data = [tref,UTS];
    dir_sel = '-';
elseif strcmp(mod,'MSD')
    r = 0;
    data = [tref,ITS-UTS];
    dir_sel = '+';
else
    sprintf('Error!')
    keyboard    
end

index = zeros(size(data));
index(:,1) = data(:,1);
if strcmp(dir_sel,'+')
    index(data(:,2)>r,2) = 1;
    index(data(:,2)<=r,2) = -1;
else
    index(data(:,2)<=r,2) = 1;
    index(data(:,2)>r,2) = -1;
end

[~,~,x0] = xlsread('IF当月.xlsx');
t0 = datenum(x0(2:end,1));
price_oc = cell2mat(x0(2:end,[2,5]));
[tref,ia,ib] = intersect(tref,t0);
index = index(ia,:);
price_oc = price_oc(ib,:);

T = length(tref);
v = zeros(T,1);
v(1) = 200000;
for i = 2:T
    if eq(index(i-1,2),-1)
        v(i) = v(i-1)+share_index_return(-1*(price_oc(i,2)-price_oc(i,1)));
    elseif eq(index(i-1,2),1)
        v(i) = v(i-1)+share_index_return(1*(price_oc(i,2)-price_oc(i,1)));
    else
        v(i) = v(i-1);
    end
    %stop_loss and takeprofit
    if v(i)/v(i-1)-1<stop_losses
        v(i) = v(i-1)*(1+stop_losses);
    end
    if v(i)/v(i-1)-1>takeprofit
        v(i) = v(i-1)*(1+takeprofit);
    end
end

figure
subplot(2,1,1);
plot(tref,index(:,2),'LineWidth',2)
datetick('x','yymm')
xlabel('时间')
ylabel('多空信号')
mark_label(gca,'A')
[sta1,sta2,sta_values] = curve_static(v);
subplot(2,1,2);
plot(tref,v/v(1),'LineWidth',2)
datetick('x','yymm')
check_data = [index,v];
xlabel('时间')
ylabel('净值曲线')
mark_label(gca,'B')

xlswrite('msd_result.xlsx',[{'时间','多空信号','曲线'};cellstr(datestr(tref,'yyyy/mm/dd')),num2cell(check_data(:,2:3))])


function share_e=share_index_return(d_index)
    share_e = d_index*300;
end

