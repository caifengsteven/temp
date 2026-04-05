%M_com_Hindenburg
%探索程序
%版本1
%20190516~20190518
clear

%参数
window_cal = 30;
window_week = 5;
%数据库
gta_astock_db = 'futuredata.STK_MKT_BWARDQUOTATION';
index_com_symbol_db = 'futuredata.a_index_composition_data';

%指数
[index_data,index_code] = get_index_data('中证全指');

index_code =['SHSE.',index_code];

index_data(:,1) = cellfun(@(x) x(1:10),index_data(:,1),'UniformOutput',false);
%[~,~,x1] = xlsread('000985_index_before.csv');
tref_str1 = index_data(:,1);
tref_str2 = fetchmysql(['select distinct tradingdate from futuredata.a_index_composition_data ',10,...
    'where index_code = ''SHSE.000985'' and tradingdate>=''2011-09-30'''],2);
[tref_str,ia,ib] = intersect(tref_str1,tref_str2);

index_data = index_data(ia,:);
close_price = cell2mat(index_data(:,end));
%y0 = cell2mat(index_data(:,end));
y0 = [0;close_price(2:end)./close_price(1:end-1)-1];
tref = datenum(index_data(:,1));


t0 = tref_str{1};
tt = tref_str{end};
symbol_all = fetchmysql(sprintf(['select distinct symbol from futuredata.STK_MKT_BWARDQUOTATION ',10,...
    'where tradingdate >=''%s'' and tradingdate<=''%s'''],t0,tt),2);

T = length(tref_str);
m = length(symbol_all);


sql_str1 = 'select symbol,closeprice/precloseprice-1 from %s where tradingdate= ''%s'' and filling = 0';
sql_str2 = 'select symbol from %s where index_code = ''%s'' and tradingdate = ''%s'' ';

load re000985_X.mat X
%{
X = nan(m,T);
计算X

%Y = cell(T,1);
load Y0
%必须并行
parfor i = 1262:T%1:T
    sub_sql = sprintf(sql_str1,gta_astock_db,tref_str{i});
    x = fetchmysql(sub_sql,2);
    sub_sql = sprintf(sql_str2,index_com_symbol_db,index_code,tref_str{i});
    y = fetchmysql(sub_sql,2);
    if strcmp(y{1}(end-4),'.')
        y = cellfun(@(x) x(1:6),y,'UniformOutput',false);
    else
        y = cellfun(@(x) x(6:end),y,'UniformOutput',false);
    end
    
    [~,ia] = intersect(x(:,1),y);
    x = x(ia,:);
    [~,ia,ib] = intersect(symbol_all,x(:,1),'stable');
    %X(ia,i) = cell2mat(x(ib,2));
    Y{i} = [ia,cell2mat(x(ib,2))];
    sprintf('%s-%d-%d',tref_str{i},i,T)
end

for i = 1:T
    temp  = Y{i};
    X(temp(:,1),i) = temp(:,2);
end
%}
%{
%合成趋同度因子
Y_pre = nan(size(X));
for i = 1:m
    sub_x = X(i,:);
    parfor j = window_cal:T
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
load re000985_Ypre Y_pre
%cal indicator
factor_v = nan(T,1);
for i = 1:T
    sub_y = Y_pre(:,i);
    sub_y(isnan(sub_y)) = [];
    if ~isempty(sub_y)
        factor_v(i) = mean(sub_y);
    end
end

yyaxis left
plot(tref,close_price,'linewidth',2);
yyaxis right
plot(tref,factor_v,'r-','LineWidth',2);

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(1:20:end),'xlim',tref([1,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);
box off
set(gca,'linewidth',1.5);


y_lim = nan(size(y0));
x_lim = y_lim;
y_lim(window_week+1:end) = y0(window_week+1:end)-y0(1:end-window_week);
x_lim(window_week+1:end) = factor_v(window_week+1:end)-factor_v(1:end-window_week);
figure;plot(x_lim,y_lim,'.')
