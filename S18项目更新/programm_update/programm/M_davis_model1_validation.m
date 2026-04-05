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
4. 上一期单季营收为正。   上一期单季净利润同比为正，
每个季度对筛选出的样本根据二阶增速降序排列，选择前 25 个样本。
回测参数设置如下：
1. 回测期间：2010 年至 2017 年 5 月 31 日；
2. 交易成本：买入 0.1%，卖出 0.2%；
3. 单只股票权重上限：10%；
4. 基准：中证 500 指数×仓位。


验证程序

%}

%利润转换为单季度利润
%计算环比

clear

%设置参数
max_symbol_num = 25;
date_cut = [1,4,7,10];
year_select = 2010:2019;
max_date = datenum(2019,5,31);

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
%获取交易日历
tref = fetchmysql('select distinct(tradingdate) from juejindata.backward_data order by tradingdate;',2);
tref_num = datenum(tref);
ind = tref_num>=tref_sel(1)&tref_num<=max_date;
tref=tref(ind);
tref_num = tref_num(ind);
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
[~,~,nin_data] = xlsread('nincome1.xlsx');
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

%格式化sql语句
sql_str_pettm =['select ticker from yuqerdata.yq_pettm where tradedate = ''%s''',...
    ' and PE<50;'];


t_sel = '20190120';
%t_sel = '20170420';
%t_sel = '20170720';
targ_symbol =xlsread('temp.xlsx',t_sel);
targ_symbol=num2cell(targ_symbol);
targ_symbol = cellfun(@(x) sprintf('%0.6d',x),targ_symbol,'UniformOutput',false);
targ_symbol0 = targ_symbol;
ind = cellfun(@(x) strcmp(x(1),'0'),targ_symbol);
targ_symbol(ind) = cellfun(@(x) [x,'.XHSE'],targ_symbol(ind),'UniformOutput',false);
targ_symbol(~ind) = cellfun(@(x) [x,'.XHSE'],targ_symbol(~ind),'UniformOutput',false);

t_sel = datenum(t_sel,'yyyymmdd');

t = find(eq(tref_sel,t_sel));
%开始选股
symbol_all = cell(T,1);
for i = t%1:T
    t1 = tref_sel(i);
    t1_trade = max(tref_num(tref_num<=t1));
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
    %当期的单季度净利润同比增速大于 20%且小于 100%，在其中根据单季增速降序排列
    %若满足条件的样本不足 25 个，则将单季增速大于 100%的样本根据单季增速升序排列，按需补充样本
    sub_nin_rate_sel = sub_nin_rate(ind_f,:);
    
    ind_sel1 = sub_nin_rate_sel(:,1)>0.2 &sub_nin_rate_sel(:,1)<1;
    sub_symbol_f1 = sub_symbol_f(ind_sel1,:);
    [~,ia] = sort(sub_nin_rate_sel(ind_sel1,1),'descend');
    sub_symbol_f1 = sub_symbol_f1(ia);
    
    ind_sel2 = sub_nin_rate_sel(:,1)>=1;
    sub_symbol_f2 = sub_symbol_f(ind_sel2,:);
    [~,ia] = sort(sub_nin_rate_sel(ind_sel2,1));
    sub_symbol_f2 = sub_symbol_f2(ia);
    
    sub_symbol_f = [sub_symbol_f1;sub_symbol_f2];
    sub_symbol_f = cellfun(@(x) x(1:6),sub_symbol_f,'UniformOutput',false);
    
    %筛选
    %剔除st
    sub_st_symbol = x_st_symbol(t1>=x_st_date0&t1<=x_st_date1);
    sub_symbol_f  = setdiff(sub_symbol_f,sub_st_symbol,'stable');
    %剔除上市不满一年
    sub_datelimit_symbol = x_listdate_symbol(t1-x_listdate_date0>365 & t1<x_listdate_date1);
    sub_symbol_f = intersect(sub_symbol_f,sub_datelimit_symbol,'stable');
    sub_symbol_f0 = sub_symbol_f;
    sub_symbol_f = sub_symbol_f(1:min(max_symbol_num,end));
    %记录
    symbol_all{i} = sub_symbol_f;
    sprintf('%d-%d',i,T)
    
    
    tes_re = cell(5,1);
    tes_re{1} = cellfun(@(x) x(1:6), sub_symbol(ind1),'UniformOutput',false);
    tes_re{2} = cellfun(@(x) x(1:6), sub_symbol(ind2),'UniformOutput',false);
    tes_re{3} = cellfun(@(x) x(1:6), sub_symbol(ind3),'UniformOutput',false);
    
    %低估值限制
    sub_pe_limit_symbol = fetchmysql(sprintf(sql_str_pettm,datestr(t1_trade,'yyyy-mm-dd')),2);
    sub_pe_limit_symbol = cellfun(@(x) sprintf('%0.6d',x),sub_pe_limit_symbol,'UniformOutput',false);
    tes_re{4} = sub_pe_limit_symbol;
    tes_re{5} = sub_symbol_f0;
    
    for j = 1:5
        temp = tes_re{j};
        temp1 = intersect(temp,targ_symbol0);
        sprintf('step %d 相同的有%d 个',j,length(temp1))
    end
    
    
    
    
end
intersect(targ_symbol0,sub_symbol_f)


