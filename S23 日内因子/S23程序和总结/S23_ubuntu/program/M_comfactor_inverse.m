%com inverse factor
clear

%parameters
dn = 'S23';
tn = 's23_factor_inverse';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_datasource = 'yuqerdata.yq_dayprice';

window_N = 20;
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
var_info = {'symbol','tradingdate','f_l','f_h','f_val'};
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

%symbols
sql_str = 'select distinct(symbol) from %s order by symbol';
symbol = fetchmysql(sprintf(sql_str,tn_datasource),2);
if istable(symbol)
    symbol = table2cell(symbol);
end
%for symbol do sth
sql_str_f1 = ['select tradedate,turnovervalue/dealamount,chgPct from %s  where symbol = ''%s'' ',...
    ' order by tradedate'];
%write to table
T = length(symbol);

parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_datasource,symbol{i}),2);
    if istable(sub_x)
        sub_x = table2cell(sub_x);
    end
    sub_x_v = cell2mat(sub_x(:,2:3));
    %remove nan
    ind_sel = ~isnan(sum(sub_x_v,2));
    sub_x = sub_x(ind_sel,:);
    sub_x_v = sub_x_v(ind_sel,:);
    
    sub_T = length(sub_x_v);
    sub_f = nan(sub_T,3);
    for j = window_N:sub_T
        sub_window = j-window_N+1:j;
        sub_sub_x = sub_x_v(sub_window,:);
        [~,ia] = sort(sub_sub_x(:,1));
        sub_f(j,1:2) = [sum(sub_sub_x(ia(1:window_N/2),2)),...
            sum(sub_sub_x(ia(window_N/2+1:end),2))];
    end
    
    if isempty(sub_f)
        continue
    end
    
    sub_f(:,3) = sub_f(:,2)-sub_f(:,1);
  
    del_ind = isnan(sub_f(:,3));
    sub_f(del_ind,:) = [];
    sub_x(del_ind,:) = [];
    

    if ~isempty(sub_f)
        sub_symbol_data = [sub_x(:,[1,1]),num2cell(sub_f)];
        sub_symbol_data(:,1) = symbol(i);
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_symbol_data)
        close(conna)
    end
    
    sprintf('%d-%d',i,T)
    
    
end