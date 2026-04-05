%{
1.细分因子标准化。对细分价值因子进行分位数变换标准化。
2.合成价值风格因子。将标准化之后的细分因子等权合成价值风格因子。
3.价值风格因子 中性化。对价值风格因子进行行业和市值的中性化处理。
%}

clear

%tref = fetchmysql('select distinct(tradeDate) from yuqerdata.yq_dayprice order by tradeDate',2);

print_sel = 0;

load tref
tref_num = datenum(tref);

sel_ind = tref_num>=datenum(2005,1,1);
tref = tref(sel_ind);
tref_num = tref_num(sel_ind);
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

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


T = length(month_cut_date2);
sql_str = 'select symbol,f_val from S30.f%d_month where tradingdate=''%s'' and f_val is not null';
sql_str_f2 = 'select symbol,f_val from S30.mv_month where tradingdate=''%s'' and f_val is not null';
symbolpool = cell(T,1);
symbolpool_f = cell(T,1);
parfor i = 1:T-1
    
    warning('off');
    
    tref_sec = month_cut_date2{i};
    tref_sec_num = datenum(tref_sec);
    %横截面数据
    for j = 1:5
        sub_x = fetchmysql(sprintf(sql_str,j,tref_sec),2);
        if eq(j,1)
            x = sub_x;
        else
            [~,ia,ib] = intersect(x(:,1),sub_x(:,1));
            x = [x(ia,:),sub_x(ib,end)];
        end
    end
    
    %st等数据
    sub_st_symbol = x_st_symbol(tref_sec_num>=x_st_date0&tref_sec_num<=x_st_date1);
    sub_st_symbol = cellfun(@(x) sprintf('%0.6d',x),sub_st_symbol,'UniformOutput',false);
    [~,ia] = intersect(x(:,1),sub_st_symbol);
    x(ia,:) = [];
    
    %因子标准化 分位数标准化
    f = cell2mat(x(:,2:end));
    for j = 1:5
        sub_x = f(:,j);
        [~,~,sub_x] = unique(sub_x);
        f(:,j) = zscore(sub_x);
    end
    %合成价值风格因子。将标准化之后的细分因子等权合成价值风格因子。
    f = mean(f,2);
    %价值风格因子 中性化。对价值风格因子进行行业和市值的中性化处理。
    indus_code = yq_methods.get_industry_class(tref_sec);
    market_value = fetchmysql(sprintf(sql_str_f2,tref_sec),2);
    inds = suscc_intersect({x(:,1),indus_code(:,1),market_value(:,1)});
    sub_symbol = x(inds(:,1),1);
    f = f(inds(:,1),1);
    sub_indus_code = cell2mat(indus_code(inds(:,2),2));
    sub_market_value = cell2mat(market_value(inds(:,3),2));
    
    %dummy
    sub_code_v_u = unique(sub_indus_code);
    dummy_v = zeros(length(sub_indus_code),length(sub_code_v_u));
    for j = 1:length(sub_code_v_u)
        dummy_v(eq(sub_indus_code,sub_code_v_u(j)),j) = 1;
    end
    %中性化   
    [~,~,r] = regress(f,[ones(size(f)),dummy_v,sub_market_value]);
    %save
    [r,ia] = sort(r);
    symbolpool{i+1} = sub_symbol(ia);
    symbolpool_f{i+1} = [f(ia),r];
    if print_sel>0
        sprintf('symbol: %d-%d',i,T)
    end
end

%分组计算
cut_num = 10;
sql_str = 'select ticker,chgPct from yuqerdata.MktEqumAdjAfGet where endDate = ''%s''';
sql_str_day = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradeDate = ''%s''';
Y = cell(T,1);
IC = cell(T,1);
parfor i = 2:T
    tref_sec = month_cut_date2{i};
    x = fetchmysql(sprintf(sql_str,tref_sec),2);
    sub_symbol = symbolpool{i};
    sub_f = symbolpool_f{i};
    [~,ib,ia] = intersect(sub_symbol,x(:,1),'stable');
    r = cell2mat(x(ia,2));
    f = sub_f(ib,:);
    IC{i} = corr(r,f,'type','Spearman')';
    sub_y = zeros(cut_num,1);
    sub_l = length(r)/cut_num;
    for j = 1:cut_num
        if j<cut_num
            sub_ind = floor((j-1)*sub_l)+1:floor(j*sub_l);
        else
            sub_ind = floor((j-1)*sub_l)+1:length(r);
        end
        sub_y(j) = mean(r(sub_ind));
    end
    Y{i} = sub_y;
    if print_sel>0
        sprintf('return: %d-%d',i,T)  
    end
end
IC = [IC{:}]';
y_r = [Y{:}]';

figure
subplot(2,1,1)
r_f = y_r(:,end)-y_r(:,1);
yyaxis left
plot(cumprod(1+r_f),'LineWidth',3)
yyaxis right
bar(r_f);

set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
box off

title('价值因子历史表现-分组多空收益')

subplot(2,1,2)
bar(IC)
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = month_cut_date2(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420*2]);
title('价值因子历史表现-IC')
movegui(gcf,'center')

[~,~,~,temp] = ttest(IC);
L = mean(cellfun(@length,symbolpool));
IC_re = [mean(IC);std(IC);min(IC);max(IC);temp.tstat;[L,L];sum(IC>0)];
IC_re = [{'','IC','adj_IC'};{'平均','标准差','最小值','最大值','t值','平均股票数','有效期数'}',num2cell(IC_re)];
disp(IC_re)
%%%%%%%%%%%%%%%%今年以来行情
ind_sel2019 = tref_num>=datenum(2018,12,28);
tref_2019 = tref(ind_sel2019);
T = length(tref_2019);
y_2019= zeros(T,cut_num);

symbolpool2019 = cell(T,1);
sub_symbol = [];
for i = 1:T
    test_ind = strcmp(month_cut_date2,tref_2019(i));
    if any(test_ind)
        sub_symbol = symbolpool{test_ind};
    end
    symbolpool2019{i} = sub_symbol;
end

parfor i = 1:T
    sub_symbol = symbolpool2019{i};

    tref_sec = tref_2019{i};
    x = fetchmysql(sprintf(sql_str_day,tref_sec),2);
    [~,~,ia] = intersect(sub_symbol,x(:,1),'stable');
    r = cell2mat(x(ia,2));
    sub_y = zeros(cut_num,1);
    sub_l = length(r)/cut_num;
    for j = 1:cut_num
        if j<cut_num
            sub_ind = floor((j-1)*sub_l)+1:floor(j*sub_l);
        else
            sub_ind = floor((j-1)*sub_l)+1:length(r);
        end
        sub_y(j) = mean(r(sub_ind));
    end
    y_2019(i,:) = sub_y';
    if print_sel>0
        sprintf('return: %d-%d',i,T)    
    end
end

r_f2019 = y_2019(:,end)-y_2019(:,1);

figure
yyaxis left
plot(cumprod(1+r_f2019),'LineWidth',3)
yyaxis right
bar(r_f2019);

set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));
t_str = tref_2019(floor(linspace(1,T,15)));
set(gca,'XTickLabel',t_str);
set(gca,'XTickLabelRotation',90)
setpixelposition(gcf,[223,365,1345,420]);
box off
title('价值因子历史表现-2019年')