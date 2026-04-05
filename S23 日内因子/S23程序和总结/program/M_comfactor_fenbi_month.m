%com inverse factor
clear

%parameters
dn = 'S23';
tn = 'fenbifactor1_month';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_source = 'S23.fenbifactor1_zscore';

%check table
%check database

sub_sql = 'show databases';
info = fetchmysql(sub_sql,2);
if istable(info)
    info = table2cell(info);
end
if ~any(strcmpi(info,dn))
    exemysql(sprintf('create database %s',dn));
end
sub_sql = sprintf('show tables from %s',dn);
info = fetchmysql(sub_sql,2);
if istable(info)
    info = table2cell(info);
end
%create database ? table ? primary key ? MyISAM
var_info = {'symbol','tradingdate','f_val'};
if ~any(strcmpi(info,tn))
    %create table    
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:2) = {'varchar(6)','date'};
    obj = mysqlTool();
    sqlquery1=obj.createTable(dn,tn,var_info,var_type);
    OK1 = exemysql(sqlquery1);
    OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dn,tn));
    OK3 = exemysql(sprintf('alter table %s.%s add primary key(symbol,tradingdate);',dn,tn));
end

%tradingdate
tref = fetchmysql(sprintf('select distinct(tradingdate) from %s order by tradingdate',tn_source),2);
if istable(tref)
    tref = table2cell(tref);
end
tref_num = datenum(tref);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
T_month_cut = size(month_cut,1);
month_cut_info = cell(T_month_cut,1);
for i = 1:T_month_cut
    month_cut_info{i} = tref(month_cut(i,1):month_cut(i,2));
end
month_cut_date = tref(month_cut(:,2));
%symbol
sql_str = 'select distinct(symbol) from %s order by symbol';
symbol = fetchmysql(sprintf(sql_str,tn_source),2);
if istable(symbol)
    symbol = table2cell(symbol);
end
%write to table
T = length(symbol);

sql_str_f1 = 'select tradingdate,f_val from %s where symbol=''%s'' order by tradingdate';

parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_source,symbol{i}),2);
    if isempty(sub_x)
        continue
    end
    if istable(sub_x)
        sub_x = table2cell(sub_x);
    end
    
    sub_y = nan(T_month_cut,1);
    for j = 1:T_month_cut
        [~,ia] = intersect(sub_x(:,1),month_cut_info{j});
        if ~isempty(ia)
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
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_symbol_data)
        close(conna)
    end
    
    sprintf('%d-%d',i,T)
    
    
end
