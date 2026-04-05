%com inverse factor
clear
key_str = 'S23分笔因子zscore';
%parameters
dn = 'S23';
tn = 'fenbifactor1_zscore';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_source = 'S23.fenbifactor1';
var_info = {'symbol','tradingdate','f_val'};
%create table    
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
[OK1,OK2,OK3] = create_table_adair(dn,tn,var_info,var_type,strjoin(var_info(1:2)));    
sql_str = 'select tradingdate from %s order by tradingdate desc limit 1';
t0 = fetchmysql(sprintf(sql_str,tn_fullname),2);
tt = 'select tradingdate from %s order by tradingdate desc limit 1';
tt = fetchmysql(sprintf(tt,tn_source),2);
if isempty(t0)
    t0 = {'2001-01-01'};
end
tref = yq_methods.get_tradingdate(t0{1},tt{1});
tref = tref(2:end);
%write to table
T = length(tref);
if eq(T,0)
    sprintf('%s无数据更新',key_str)
    return
end
sql_str_f1 = 'select symbol,tradingdate,spread_date_adj from %s where tradingdate=''%s''';
parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_source,tref{i}),2);
    if istable(sub_x)
        sub_x = table2cell(sub_x);
    end
    sub_x_v = cell2mat(sub_x(:,3));
    %remove nan
    ind_sel = ~isnan(sum(sub_x_v,2));
    sub_x = sub_x(ind_sel,:);
    sub_x_v = sub_x_v(ind_sel,:);
    
    if isempty(sub_x_v)
        continue
    end
    
    sub_x_v = zscore(sub_x_v);
    sub_f = [sub_x(:,1:2),num2cell(sub_x_v)];

    if ~isempty(sub_f)
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_f)
        close(conna)
    end
    
    sprintf('%s:%d-%d',key_str,i,T)
    
    
end