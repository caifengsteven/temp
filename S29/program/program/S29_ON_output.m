%{
S29结果输出
%}
clear
close all

tt = datestr(now,'yyyy-mm-dd');
tref1 = yq_methods.get_tradingdate('2013-01-01',tt);
tref2 = yq_methods.get_tradingdate_future(tref1{end});
tref = [tref1;tref2(2)];
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

index_pool = {[],'000905','000300'};
index_info = {'全市场','中证500','沪深300'};
factor_name = {'双向选择','交叉不限制行业','交叉限制行业'};

t_sel = month_cut_date2{end};

re1 = cell(3,1);
re2 = re1;
re3 = re1;
for i = 1:3
    index_sel = index_pool{i};
    
    symbol_pool1 = S29_sel_symbol.get_symbol_s1(F_all,x_st_symbol,x_st_date0,x_st_date1,t_sel,index_sel); %原始做法
    symbol_pool3 = S29_sel_symbol.get_symbol_s3(F_all,x_st_symbol,x_st_date0,x_st_date1,t_sel,index_sel); %交叉，不限制行业内
    symbol_pool4 = S29_sel_symbol.get_symbol_s4(F_all,x_st_symbol,x_st_date0,x_st_date1,t_sel,index_sel); %交叉，限制行业

    symbol_pool1 = symbol_pool1(:,[1,1]);
    symbol_pool1(:,1) = index_info(i);
    
    symbol_pool3 = symbol_pool3(:,[1,1]);
    symbol_pool3(:,1) = index_info(i);

    symbol_pool4 = symbol_pool4(:,[1,1]);
    symbol_pool4(:,1) = index_info(i);
    re1(i) = {symbol_pool1'};
    re2(i) = {symbol_pool3'};
    re3(i) = {symbol_pool4'};
end
re1 = [re1{:}]';
re1 =[factor_name([1,1]);re1];
re2 = [re2{:}]';
re2 =[factor_name([2,2]);re2];
re3 = [re3{:}]';
re3 =[factor_name([3,3]);re3];

max_L = max([size(re1,1),size(re2,1),size(re3,1)]);
re = cell(max_L,6);
re(1:size(re1,1),1:2) = re1;
re(1:size(re2,1),3:4) = re2;
re(1:size(re3,1),5:6) = re3;

gui_result(re(2:end,:),t_sel,re(1,:));

re = cell2table(re);
writetable(re,sprintf('S29_%s选股结果.csv',t_sel))
