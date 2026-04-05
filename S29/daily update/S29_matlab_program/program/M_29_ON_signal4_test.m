%{
每一期每个行业内选取排名处于前 1/10 分位数以上的股票，行业内等权。
选取中证 500 作为基准，行业间权重按照指数行业权重配比。
策略月换仓。
交易手续费考虑双边千三，剔除 ST、涨跌停等交易受到限制的股票。

流程
0 static return of last month?
1 data: month return
2 data: factor and factor ttm
3 data: industry class - dummy variable
4 data: st return limit
5 caculate and get symbol pool and update symbol pool

mod = 1 反转
mod = 2 动量
update 
1限制500股票池
3行业内排名，行业内交叉选股
4 限制行业选股，股票池为500，选股按照排名后，按照排序取top10%，然后交叉选择
%}
clear
close all

tref = yq_methods.get_tradingdate('2013-01-01',datestr(now,'yyyy-mm-dd'));
%找到月底最后一天
tref_num = datenum(tref);
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

%读入因子数据
sql_str_f1 = ['select factor_name,pub_date,symbol,f_val from ',...
    'S29.factor_yuqer_com order by pub_date desc']; 
sql_str_f2 = ['select factor_name,pub_date,symbol,f_val from ',...
    'S29.factor_yuqer_com_ttm order by pub_date desc'];

sql_str_f3 = ['select ticker,chgPct from yuqerdata.MktEqumAdjAfGet where ',...
    'endDate=''%s'' and chgPct is not null'];

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

T = length(month_cut_date2);
y_all = zeros(T,1);

symbol_history = cell(T,1);

index_sel = '000905';
for i = 1:T    
    
    sub_t_num = datenum(month_cut_date2{i});
    %月度数据
    x = fetchmysql(sprintf(sql_str_f3,month_cut_date2{i}),2);
    sub_symbol_pool = get_index_com_symbol(index_sel,month_cut_date2{i});
    [~,ia] = intersect(x(:,1),sub_symbol_pool);
    x = x(ia,:);
    sub_y = cell2mat(x(:,2));
    if i > 1
        temp = zeros(size(symbol_pool));
        [~,ia,ib] = intersect(symbol_pool,x(:,1));
        temp(ia) = sub_y(ib);
        y_all(i) = mean(temp)-3/1000;
        %y_all(i,j) = mean(sub_y(ib));
        %%%%%%%%%   
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
    sub_code = yq_methods.get_industry_class(month_cut_date2{i});
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
    sub_indus_code = sub_code_v(~nan_ind,:);
    
    %linner regression
    [~,~,r] = regress(sub_y,[ones(size(sub_y)),sub_x]); 
    [~,~,ia] = unique(r);
    [~,~,ia2] = unique(-(sub_y-r));
    
    sub_indus_code_u = unique(sub_indus_code);
    ia_f = [];
    for j = 1:length(sub_indus_code_u)
        sub_ind = find(eq(sub_indus_code,sub_indus_code_u(j)));
        [~,sub_ia] = sort(ia(sub_ind));
        [~,sub_ia2] = sort(ia2(sub_ind));
        k = 1;
        while k < 10
            sub_ia_f = intersect(sub_ia(1:ceil(end/10*k)),sub_ia2(1:ceil(end/10*k)));
            k = k + 1;
            if ~isempty(sub_ia_f)
                k = 20;
            end
        end
        ia_f = cat(1,ia_f,sub_ind(sub_ia_f));
    end
    
    symbol_pool = sub_symbol(ia_f);
    temp = symbol_pool(:,[1,1]);
    temp(:,1) = month_cut_date2(i);
    symbol_history{i} = temp;
    sprintf('%d-%d',i,T)
end

symbol_history=cellfun(@(x) x',symbol_history,'UniformOutput',false);
symbol_history = [symbol_history{:}]';

y_index = fetchmysql('SELECT endDate,chgPct FROM yuqerdata.yq_index_month where symbol = ''000905'' order by endDate;',2);
[~,~,ib] = intersect(month_cut_date2,y_index(:,1),'stable');
y_index_value = cell2mat(y_index(ib,2));
y_index_value(1) = 0;

figure
yyaxis left
plot(cumprod(1+(y_all-y_index_value)),'LineWidth',3)
yyaxis right
bar((y_all-y_index_value));

set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420]);
box off



