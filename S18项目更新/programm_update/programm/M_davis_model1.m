%{
戴维斯双击选股模型的构建

一
4月20,7月20日、10月20日和1月20日作为调仓时点
剔除 ST 和上市不满 1 年的新股。

二
1. 单季净利润的同比增速为正且前期净利润大于 300 万；
2. 上一期单季净利润的同比增速为正；
3. 分别计算当期单季净利润的同比增速和上一期单季净利润的同比增速，并对 2 个增速
环比计算二阶增速，要求二阶增速为正，即 2 个季度加速增长；
4. 上一期单季营收为正。
每个季度对筛选出的样本根据二阶增速降序排列，选择前 25 个样本。
回测参数设置如下：
1. 回测期间：2010 年至 2017 年 5 月 31 日；
2. 交易成本：买入 0.1%，卖出 0.2%；
3. 单只股票权重上限：10%；
4. 基准：中证 500 指数×仓位。

%}

%利润转换为单季度利润
%计算环比

clear

%设置参数
max_symbol_num = 25;
date_cut = [1,4,7,10];
year_select = 2010:2017;
max_date = datenum(2017,5,31);

%获取所有选股时间节点
tref_sel = zeros(length(year_select)*length(date_cut),1);
k = 0;
for i = 1:length(year_select)
    for j = 1:length(date_cut)
        temp_t = datenum(year_select(i),date_cut(j),20);
        k = k + 1;
        tref_sel(k) = temp_t;
    end
end
tref_sel(tref_sel>max_date) = [];
T = length(tref_sel);
%载入ST信息数据
sql_str = 'SELECT * FROM yuqerdata.st_info order by tradedate desc';
x_st = fetchmysql(sql_str,2);
x_st_codenum = cell2mat(x_st(:,1));
x_st_u_codenum = unique(x_st_codenum);
x_st_data = cell(length(x_st_u_codenum),3);
for i = 1:length(x_st_u_codenum)
    sub_x_st_data=x_st(eq(x_st_codenum,x_st_u_codenum(i)),:);
    x_st_data(i,:) = {sprintf('%0.6d',x_st_u_codenum(i)),sub_x_st_data{1,2},sub_x_st_data{end,2}};
end
x_st_symbol = x_st_data(:,1);
x_st_date0 = datenum(x_st_data(:,3));
x_st_date1 = datenum(x_st_data(:,2));
%释放内存
clear x_st x_st_codenum x_st_u_codenum

%载入上市时间数据
sql_str = ['SELECT ticker,listDate,delistDate FROM yuqerdata.stock_basic_info ',...
    'where (ticker like ''0%'' or ticker like ''6%'' or ticker like ''3%'');'];
x_listdate = fetchmysql(sql_str,2);
del_ind = strcmp(x_listdate(:,2),'null');
x_listdate(del_ind,:) = [];
c_ind = strcmp(x_listdate(:,3),'null');
x_listdate(c_ind,3) = {'9999-12-31'};
x_listdate_symbol = x_listdate(:,1);
x_listdate_date0 = datenum(x_listdate(:,2));
x_listdate_date1 = datenum(x_listdate(:,3));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%载入净利润数据
[~,~,nin_data] = xlsread('nincome.xlsx');
nin_data =nin_data(2:end,:);
%数据按照从新-旧排序，便于后续检索
nin_data = flipud(nin_data);

%去除缺失数据
nan_ind = cellfun(@isnan,nin_data(:,7));
nin_data(nan_ind,:) = [];

%获取净利润数据
nin_symbol = nin_data(:,1);
nin_endDate = datenum(nin_data(:,2));
nin_pubDate = datenum(nin_data(:,3));
nin_v = cell2mat(nin_data(:,[4,6,8]));
nin_rate = cell2mat(nin_data(:,[5,7]));

%开始选股
symbol_all = cell(T,1);
for i = 1:T
    t1 = tref_sel(i);
    ind1 = nin_pubDate<=t1;
    sub_symbol_all = nin_symbol(ind1);
    sub_endDate=nin_endDate(ind1);
    sub_pubDate=nin_pubDate(ind1);
    sub_nin_v = nin_v(ind1,:);
    sub_nin_rate = nin_rate(ind1,:);
    [sub_symbol,ia] = unique(sub_symbol_all,'stable');
    sub_endDate=sub_endDate(ia);
    sub_pubDate=sub_pubDate(ia);
    sub_nin_v = sub_nin_v(ia,:);
    sub_nin_rate = sub_nin_rate(ia,:);
    %%%筛选
    %1. 单季净利润的同比增速为正且前期净利润大于 300 万； %前期净利润时同比时用的净利润还是上一期的净利润？
    ind1 = sub_nin_rate(:,1)>0 & sub_nin_v(:,3)>300*10000 &sub_nin_v(:,1)>300*10000;
    %2. 上一期单季净利润的同比增速为正；
    ind2 = sub_nin_rate(:,2)>0;
    %3. 分别计算当期单季净利润的同比增速和上一期单季净利润的同比增速，并对 2 个增速
    %环比计算二阶增速，要求二阶增速为正，即 2 个季度加速增长；
    ind3 = (sub_nin_rate(:,1)-sub_nin_rate(:,2))>0;
    % 4. 上一期单季营收为正。
    ind4 = sub_nin_v(:,2)>0;
    %合并条件
    ind_f = ind1 & ind2 & ind3 & ind4;
    sub_symbol_f = sub_symbol(ind_f);
    %排序
    sub_ind_value = (sub_nin_rate(ind_f,1)-sub_nin_rate(ind_f,2))./sub_nin_rate(ind_f,2);
    [~,ia] = sort(sub_ind_value,'descend');
    sub_symbol_f = sub_symbol_f(ia,:);
    sub_symbol_f = cellfun(@(x) x(1:6),sub_symbol_f,'UniformOutput',false);
    %筛选
    %剔除st
    sub_st_symbol = x_st_symbol(t1>=x_st_date0&t1<=x_st_date1);
    sub_symbol_f  = setdiff(sub_symbol_f,sub_st_symbol,'stable');
    %剔除上市不满一年
    sub_datelimit_symbol = x_listdate_symbol(t1-x_listdate_date0>365 & t1<x_listdate_date1);
    sub_symbol_f = intersect(sub_symbol_f,sub_datelimit_symbol,'stable');
    sub_symbol_f = sub_symbol_f(1:min(max_symbol_num,end));
    %记录
    symbol_all{i} = sub_symbol_f;
    sprintf('%d-%d',i,T)
end

%开始回测
%获取交易日历
tref = fetchmysql('select distinct(tradingdate) from juejindata.backward_data order by tradingdate;',2);
tref_num = datenum(tref);
ind = tref_num>=tref_sel(1)&tref_num<=datenum(2019,7,1);
tref=tref(ind);
tref_num = tref_num(ind);

T = length(tref_sel);
y = zeros(size(tref_num));
%初始权重
ini_v = 1/max_symbol_num;
for i = 1:T
    if i < T
        sub_t = [tref_sel(i)+1,tref_sel(i+1)];
    else
        sub_t = [tref_sel(i)+1,tref_num(end)];
    end
    sub_symbol = symbol_all{i};
    
    t_ind_sel = tref_num>=sub_t(1)&tref_num<=sub_t(2);
    sub_tref = tref(t_ind_sel);
    sub_y  = get_single_bac_return(sub_symbol,sub_t,sub_tref);
    
    sub_y_a = sum(ini_v*cumprod(1+sub_y),2);    
    ini_v = sub_y_a(end)/max_symbol_num;
    
    y(t_ind_sel) = sub_y_a;
    
    sprintf('%d-%d',i,T)
end

ind = tref_num<=datenum(2017,5,31);
sub_t = tref_num(ind);
sub_tref = tref(ind);
sub_y = y(ind);
sub_y(1) = 1;

x_ref = fetchmysql(['SELECT tradingdate,close FROM futuredata.indicator_data ',...
    'where symbols=''SHSE.000905'' order by tradingdate;'],2);

[~,ia,ib] = intersect(x_ref(:,1),sub_tref);
sub_y_ref = cell2mat(x_ref(ia,2));
sub_y = sub_y(ib);

sub_y_tref = sub_y_ref/sub_y_ref(1);
figure
plot([sub_y,sub_y_tref,sub_y-sub_y_tref+1],'LineWidth',2);

sub_x = sub_tref(ib);
sub_ind = floor(linspace(1,length(sub_y),20));
set(gca,'xtick',sub_ind);
set(gca,'xlim',[1,length(sub_y)]);
set(gca,'XTickLabel',sub_x(sub_ind));
set(gca,'XTickLabelRotation',45);
leg_strs = {'策略','中证500*持仓','相对强弱'};
legend(leg_strs,'location','northwest','NumColumns',length(leg_strs))



function sub_y  = get_single_bac_return(sub_symbol,sub_t,sub_tref)
    fee1 = 1/1000;
    fee2 = 2/1000;
    sql_str_m1 = ['select symbol,tradingdate,close/pre_close-1 from juejindata.backward_data where symbol in(%s) ',...
    'and tradingdate>=''%s'' and tradingdate<=''%s'';'];

    ind_0 = cellfun(@(x) strcmpi(x(1),'0'),sub_symbol);
    sub_symbol(ind_0) = cellfun(@(x) ['SZSE.',x],sub_symbol(ind_0),'UniformOutput',false);
    sub_symbol(~ind_0) = cellfun(@(x) ['SHSE.',x],sub_symbol(~ind_0),'UniformOutput',false);
    
    
    sub_str1 = sprintf('''%s''',strjoin(sub_symbol,''','''));
    sub_str2 = datestr(sub_t(1),'yyyy-mm-dd');
    sub_str3 = datestr(sub_t(2),'yyyy-mm-dd');
    
    sub_x = fetchmysql(sprintf(sql_str_m1,sub_str1,sub_str2,sub_str3),2);
    
    sub_y = zeros(length(sub_tref),length(sub_symbol));
    for j = 1:length(sub_symbol)
        sub_sub_x = sub_x(strcmp(sub_x(:,1),sub_symbol(j)),:);
        [~,ia] = intersect(sub_tref,sub_sub_x(:,2),'stable');
        sub_y(ia,j) = cell2mat(sub_sub_x(:,3));
    end
    sub_y(1,:) = sub_y(1,:)-fee1;%买入
    sub_y(end,:) = sub_y(end,:) - fee2;%卖出
end




