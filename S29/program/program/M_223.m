%{
2.2.1. 投机性收益的反转属性
本文采用分层测试的方法，
每个横截面将全部股票按照投机性收益从高到低的顺序在行业内部等
分成五组，构建五个股票组合并持有到下一个换仓日，也即下个月末
年报发布时间
1季度 4/30
2     8/30
3     10/31
4     下一年4/30
流程
0 static return of last month?
1 data: month return
2 data: factor and factor ttm
3 data: industry class - dummy variable
4 data: st return limit
5 caculate and get symbol pool and update symbol pool

mod = 1 反转
mod = 2 动量
update 同时计算出反转和动量
%}
clear
close all

cut_num = 5;%分组个数
method_name = cell(cut_num,1);
for i = 1:cut_num
    method_name{i} = sprintf('G%d',i);
end
tref = yq_methods.get_tradingdate('2008-07-01','2019-09-31');
%找到月底最后一天
tref_num = datenum(tref);
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

%读入因子数据
sql_str_f1 = ['select factor_name,pub_date,symbol,f_val from ',...
    'S29.factor_wind_com order by pub_date desc']; 
sql_str_f2 = ['select factor_name,pub_date,symbol,f_val from ',...
    'S29.factor_wind_com_ttm order by pub_date desc'];

sql_str_f3 = ['select ticker,chgPct from yuqerdata.MktEqumAdjAfGet where ',...
    'endDate=''%s'''];

%载入ST信息数据
sql_str = 'SELECT * FROM yuqerdata.st_info order by tradedate desc';
x_st = fetchmysql(sql_str,2);
x_st(:,1) = cellfun(@str2double,x_st(:,1),'UniformOutput',false);
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

%{
F = fetchmysql(sql_str_f1,2);
[temp,~,ib] = unique(F(:,2));
temp = datenum(temp);
t_F =temp(ib);
F_ttm = fetchmysql(sql_str_f2,2);
[temp,~,ib] = unique(F_ttm(:,2));
temp = datenum(temp);
t_F_ttm =temp(ib);

F_all = cell(10,2);
for i = 1:5
    sub_ind1 = strcmp(F(:,1),sprintf('cF%d',i));
    F_all{i,1} = F(sub_ind1,:);
    F_all{i,2} = t_F(sub_ind1,:);
    
    sub_ind2 = strcmp(F_ttm(:,1),sprintf('ctm%d',i));
    F_all{i+5,1} = F_ttm(sub_ind2,:);
    F_all{i+5,2} = t_F_ttm(sub_ind2,:);
    
end
%}
load F_all F_all

symbol_pool = cell(cut_num,1);
symbol_pool_2 = symbol_pool;
symbol_pool_c = symbol_pool;
T = length(month_cut_date2);

y_all = zeros(T,cut_num);
y_all2 = y_all;
y_allc = y_all;
for i = 1:T    
    sub_t_num = datenum(month_cut_date2{i});
    %月度数据
    x = fetchmysql(sprintf(sql_str_f3,month_cut_date2{i}),2);
    sub_y = cell2mat(x(:,2));
    if i > 1
        for j = 1:cut_num
            temp = zeros(size(symbol_pool{j}));
            [~,ia,ib] = intersect(symbol_pool{j},x(:,1));
            temp(ia) = sub_y(ib);
            y_all(i,j) = mean(temp);
            %y_all(i,j) = mean(sub_y(ib));
            %%%%%%%%%
            temp = zeros(size(symbol_pool_2{j}));
            [~,ia,ib] = intersect(symbol_pool_2{j},x(:,1));
            temp(ia) = sub_y(ib);
            y_all2(i,j) = mean(temp);
            %y_all2(i,j) = mean(sub_y(ib));
            
            temp = zeros(size(symbol_pool_c{j}));
            [~,ia,ib] = intersect(symbol_pool_c{j},x(:,1));
            temp(ia) = sub_y(ib);
            y_allc(i,j) = mean(temp);
            %y_allc(i,j) = mean(sub_y(ib));
            
        end
    end
    %每个因子数据
    sub_F = nan(size(x,1),10);
    for j = 1:10
        sub_f = F_all{j,1}(F_all{j,2}<=sub_t_num,:);
        [~,ia] =unique(sub_f(:,3),'stable');
        sub_f = sub_f(ia,[3,4]);
        [~,ia,ib] = intersect(x(:,1),sub_f(:,1),'stable');
        sub_F(ia,j) = cell2mat(sub_f(ib,2));        
    end    
    %行业数据
    sub_code = yq_methods.get_industry_class(tref{i});
    [~,ia,ib] = intersect(x(:,1),sub_code(:,1),'stable');
    sub_code_v = zeros(size(x(:,1)));
    sub_code_v(ia) = cell2mat(sub_code(ib,2));
    %dummy
    sub_code_v_u = unique(sub_code_v);
    dummy_v = zeros(length(sub_code_v),length(sub_code_v_u));
    for j = 1:length(sub_code_v_u)
        dummy_v(eq(sub_code_v,sub_code_v_u(j)),j) = 1;
    end
    %st等数据
    sub_st_symbol = x_st_symbol(sub_t_num>=x_st_date0&sub_t_num<=x_st_date1);
    sub_st_symbol = cellfun(@(x) sprintf('%0.6d',x),sub_st_symbol,'UniformOutput',false);
    [~,del_ind] = intersect(x(:,1),sub_st_symbol,'stable');
    %涨跌停
    %综合数据 y = kx + b
    sub_x = [sub_F,dummy_v];
    nan_ind = isnan(sum(sub_x,2)+sub_y);
    nan_ind(del_ind) = true;
    sub_y = sub_y(~nan_ind,:);
    sub_x = sub_x(~nan_ind,:);
    sub_symbol = x(~nan_ind,1);
    
    %linner regression
    [~,~,r] = regress(sub_y,[ones(size(sub_y)),sub_x]); 
    [~,ia] = sort(-r);
    [~,ia2] = sort(sub_y-r);
    %分组
    sub_num = floor(length(ia)/cut_num);
    for j = 1:cut_num
        if j < cut_num
            sub_ind = sub_num*(j-1)+1:j*sub_num;
        else
            sub_ind = sub_num*(j-1)+1:length(ia);
        end
        symbol_pool{j} = sub_symbol(ia(sub_ind));
        symbol_pool_2{j} = sub_symbol(ia2(sub_ind));
        symbol_pool_c{j} = intersect(symbol_pool{j},symbol_pool_2{j});
        
    end
    sprintf('%d-%d',i,T)
end

figure
subplot(3,1,1)
plot(cumprod(1+y_all),'LineWidth',3)
legend(method_name,'NumColumns',cut_num,'Location','northwest')
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420]);
box off

subplot(3,1,2)
plot(cumprod(1+y_all2),'LineWidth',3)
legend(method_name,'NumColumns',cut_num,'Location','northwest')
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420]);
box off

subplot(3,1,3)
plot(cumprod(1+y_allc),'LineWidth',3)
legend(method_name,'NumColumns',cut_num,'Location','northwest')
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420]);
box off


