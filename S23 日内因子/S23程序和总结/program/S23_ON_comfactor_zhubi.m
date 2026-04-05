%合并程序，一次计算完毕所有逐笔涉及因子
%帕累托
%
clear

%parameters
key_str = 'S23逐笔基础因子合成';
dn = 'S23';
tn = 'zhubifactor_pareto';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_source = 'S23.zhubifactor_basic';

%create database ? table ? primary key ? MyISAM
var_info = {'symbol','tradingdate','Bf_val','Bf_val_adj','Sf_val','Sf_val_adj'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
[OK1,OK2,OK3] = create_table_adair(dn,tn,var_info,var_type,strjoin(var_info(1:2)));
%%%%%%%%%% 大额成交因子
tn2 = 'zhubifactor_volumebig';
tn_fullname2 = sprintf('%s.%s',dn,tn2);
var_info2 = {'symbol','tradingdate','BB1','BS1','minus_BS1','sum_BS1',...
    'BB2','BS2','minus_BS2','sum_BS2','focus_B','focus_S','minus_fBS','sum_fBS'};
    %create table    
var_type2 = cell(size(var_info2));
var_type2(:) = {'float'};
var_type2(1:2) = {'varchar(6)','date'};
create_table_adair(dn,tn2,var_info2,var_type2,strjoin(var_info2(1:2)));    
%%%%%%%%%%%%%%%%大额成交占比因子
tn3 = 'zhubifactor_volumeratio';
tn_fullname3 = sprintf('%s.%s',dn,tn3);
com_num = nchoosek(1:7,2);
com_strS = cell(size(com_num(:,1)'));
com_strB = com_strS;
for i = 1:size(com_num,1)
    com_strB{i} = sprintf('Bf_val%d%d',com_num(i,1),com_num(i,2));
    com_strS{i} = sprintf('Sf_val%d%d',com_num(i,1),com_num(i,2));
end
var_info3 = [{'symbol','tradingdate'},com_strB,com_strS];
var_type3 = cell(size(var_info3));
var_type3(:) = {'float'};
var_type3(1:2) = {'varchar(6)','date'};
create_table_adair(dn,tn3,var_info3,var_type3,strjoin(var_info3(1:2)));      
    
%tradingdate
month_cut_date = yq_methods.get_month_data();
month_cut_date_num = datenum(month_cut_date);
tref = yq_methods.get_tradingdate(month_cut_date{1},month_cut_date{end});
tref_num = datenum(tref);
T_month_cut = length(month_cut_date)-1;
month_cut_info = cell(T_month_cut,1);
for i = 1:T_month_cut
    sub_id = tref_num>month_cut_date_num(i)&tref_num<=month_cut_date_num(i+1);
    month_cut_info{i} = tref(sub_id);
end
month_cut_date = month_cut_date(2:end);
month_cut_date_num = month_cut_date_num(2:end);
sql_str = 'select tradingdate from %s order by tradingdate desc limit 1';
t0_1 = fetchmysql(sprintf(sql_str,tn_fullname),2);
t0_2 = fetchmysql(sprintf(sql_str,tn_fullname2),2);
t0_3 = fetchmysql(sprintf(sql_str,tn_fullname3),2);

if isempty(t0_1)
    t0_1 = {'2001-01-01'};
end
if isempty(t0_2)
    t0_2 = {'2001-01-01'};
end
if isempty(t0_3)
    t0_3 = {'2001-01-01'};
end

id1 = month_cut_date_num>datenum(t0_1);
month_cut_date1 = month_cut_date(id1);
month_cut_info1 = month_cut_info(id1);
T_month_cut1 = length(month_cut_date1);

id2 = month_cut_date_num>datenum(t0_2);
month_cut_date2 = month_cut_date(id2);
month_cut_info2 = month_cut_info(id2);
T_month_cut2 = length(month_cut_date2);

id3 = month_cut_date_num>datenum(t0_1);
month_cut_date3 = month_cut_date(id3);
month_cut_info3 = month_cut_info(id3);
T_month_cut3 = length(month_cut_date3);
if eq(T_month_cut1,0) && eq(T_month_cut2,0) && eq(T_month_cut3,0)
    sprintf('%s已经是最新日期%s',key_str,t0_1{1})
    return
end
t0_a = [t0_1,t0_2,t0_3];
[~,ia] = min(datenum(t0_a));
t0 = t0_a(ia);

ia = find(strcmp(tref,t0));
if isempty(ia)
    ia = 1;
end
t00 = tref{max(ia-30,1)};
%symbol
symbol = yq_methods.get_symbol_A();
%write to table
T = length(symbol);

var1 = 'Bbeta,Bbeta_adj,Sbeta,Sbeta_adj';
var2 = 'bigB1,bigS1,bigB2,bigS2,focusB,focusS';
var3 = ['BV1,BV2,BV3,BV4,BV5,BV6,BV7,SV1,SV2,SV3,',...
    'SV4,SV5,SV6,SV7'];


sql_str_f1 = 'select tradingdate,%s,%s,%s from %s where symbol=''%s'' and tradingdate>=''%s'' order by tradingdate';

parfor i = 1:T
    X = fetchmysql(sprintf(sql_str_f1,var1,var2,var3,tn_source,symbol{i},t00),3);
    if isempty(X)
        continue
    end
    
    sub_x = [X.tradingdate,num2cell(X{:,strsplit(var1,',')})];    
    sub_sub_x = cell2mat(sub_x(:,2:end));
    del_ind = isnan(sum(sub_sub_x,2));
    sub_sub_x(del_ind,:) = [];
    sub_x(del_ind,:) =[];    
    sub_sub_x = movmean(sub_sub_x,[20-1,0]);    
    sub_y = nan(T_month_cut1,4);
    for j = 1:T_month_cut1
        [~,ia] = intersect(sub_x(:,1),month_cut_info1{j});
        if ~isempty(ia)
            sub_y(j,:) = sub_sub_x(ia(end),:);
        end
    end        
    sub_symbol_data1 = [month_cut_date1,month_cut_date1,num2cell(sub_y)];
    sub_symbol_data1(:,1) = symbol(i);
    ia = ~isnan(sum(sub_y,2));
    sub_symbol_data1 = sub_symbol_data1(ia,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sub_x = [X.tradingdate,num2cell(X{:,strsplit(var2,',')})]; 
    sub_sub_x = cell2mat(sub_x(:,2:end));
    del_ind = isnan(sum(sub_sub_x,2));
    sub_sub_x(del_ind,:) = [];
    sub_x(del_ind,:) =[];    
    %caculate
    sub_sub_x1 = [sub_sub_x(:,1:2),sub_sub_x(:,1)-sub_sub_x(:,2),sum(sub_sub_x(:,1:2),2)];
    sub_sub_x2 = [sub_sub_x(:,3:4),sub_sub_x(:,3)-sub_sub_x(:,4),sum(sub_sub_x(:,3:4),2)];
    sub_sub_x3 = [sub_sub_x(:,5:6),sub_sub_x(:,5)-sub_sub_x(:,6),sum(sub_sub_x(:,5:6),2)];  
    sub_sub_x = movmean([sub_sub_x1,sub_sub_x2,sub_sub_x3],[20-1,0]);
    
    sub_y = nan(T_month_cut2,size(sub_sub_x,2));
    for j = 1:T_month_cut2
        [~,ia] = intersect(sub_x(:,1),month_cut_info2{j});
        if ~isempty(ia)
            sub_y(j,:) = sub_sub_x(ia(end),:);
        end
    end
 
    sub_symbol_data2 = [month_cut_date2,month_cut_date2,num2cell(sub_y)];
    sub_symbol_data2(:,1) = symbol(i);
    ia = ~isnan(sum(sub_y,2));
    sub_symbol_data2 = sub_symbol_data2(ia,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sub_x = [X.tradingdate,num2cell(X{:,strsplit(var3,',')})]; 
    sub_sub_x = cell2mat(sub_x(:,2:end));
    del_ind = isnan(sum(sub_sub_x,2));
    sub_sub_x(del_ind,:) = [];
    sub_x(del_ind,:) =[];
    sub_sub_xb = sub_sub_x(:,1:7);
    sub_sub_xs = sub_sub_x(:,8:end);
    sub_x_com_b = zeros(size(sub_sub_x,1),size(com_num,1));
    sub_x_com_s = sub_x_com_b;
    for j = 1:size(com_num,1)
        a = com_num(j,1);
        b = com_num(j,2);
        sub_x_com_b(:,j) = sub_sub_xb(:,a)./sub_sub_xb(:,b);
        sub_x_com_s(:,j) = sub_sub_xs(:,a)./sub_sub_xs(:,b);
    end
    
    sub_sub_x = movmean([sub_x_com_b,sub_x_com_s],[20-1,0]);
    
    sub_y = nan(T_month_cut3,size(sub_sub_x,2));
    for j = 1:T_month_cut3
        [~,ia] = intersect(sub_x(:,1),month_cut_info3{j});
        if ~isempty(ia)
            sub_y(j,:) = sub_sub_x(ia(end),:);
        end
    end
    sub_symbol_data3 = [month_cut_date3,month_cut_date3,num2cell(sub_y)];
    sub_symbol_data3(:,1) = symbol(i);
    ia = ~isnan(sum(sub_y,2));
    sub_symbol_data3 = sub_symbol_data3(ia,:);
    
    %%%%%%%%%%%%%%%%%
    if ~isempty(sub_symbol_data1)
        conna = mysql_conn();
        datainsert(conna,tn_fullname,var_info,sub_symbol_data1)
        close(conna)
    end
    if ~isempty(sub_symbol_data2)
        conna = mysql_conn();
        datainsert(conna,tn_fullname2,var_info2,sub_symbol_data2)
        close(conna)
    end
    
    if ~isempty(sub_symbol_data3)
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname3,var_info3,sub_symbol_data3)
        close(conna)
    end
    sprintf('%s:%d-%d',key_str,i,T)
end
