clear
key_str = 'S23逐笔因子';
%parameters
dn = 'S23';
tn = 'fenbifactor1_month';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_source = 'S23.fenbifactor1_zscore';


var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
[OK1,OK2,OK3] = create_table_adair(dn,tn,var_info,var_type,strjoin(var_info(1:2)));    

%%%%%%%%%%
month_cut_date = yq_methods.get_month_data();
month_cut_date_num = datenum(month_cut_date);
sql_str0 = 'select tradingdate from %s order by tradingdate desc limit 1';
t0 = fetchmysql(sprintf(sql_str0,tn_fullname),2);
if isempty(t0)
    t0 = {'2010-01-01'};
end
tt = fetchmysql(sprintf(sql_str0,tn_source),2);
ind = month_cut_date_num>=datenum(t0) & month_cut_date_num<=datenum(tt);
month_cut_date = month_cut_date(ind);
month_cut_date_num = month_cut_date_num(ind);

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

T_month_cut = size(month_cut_date,1);
if isempty(month_cut_date)
    sprintf('%s无数据更新',key_str)
    return
end
%symbol
symbol = yq_methods.get_symbol_A();
%write to table
T = length(symbol);
sql_str_f1 = 'select tradingdate,f_val from %s where symbol=''%s'' and tradingdate>=''%s''  order by tradingdate';

re = cell(T,1);
parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_source,symbol{i},t0{1}),2);
    if isempty(sub_x)
        continue
    end
    if istable(sub_x)
        sub_x = table2cell(sub_x);
    end
    
    sub_y = nan(T_month_cut,1);
    for j = 1:T_month_cut
        [~,ia] = intersect(sub_x(:,1),month_cut_info{j});
        if  length(ia)>10 %一个月至少10个交易日
            sub_v = cell2mat(sub_x(ia,2));
            sub_w = (1:length(ia))'./length(ia);
            sub_y(j) = sum(sub_v.*sub_w)/sum(sub_w);
        end
    end
    
    sub_symbol_data = [month_cut_date,month_cut_date,num2cell(sub_y)];
    sub_symbol_data(:,1) = symbol(i);
    ia = ~isnan(sub_y);
    sub_symbol_data = sub_symbol_data(ia,:);

    if ~isempty(sub_symbol_data)
        re{i} = sub_symbol_data';
    end
    
    sprintf('%s:%d-%d',key_str,i,T)
    
end
re = [re{:}]';
if ~isempty(re)
    conna = mysql_conn();
    %write data to mysql
    datainsert(conna,tn_fullname,var_info,re)
    close(conna)
end