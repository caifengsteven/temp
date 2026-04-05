%com inverse factor
clear

%parameters
dn = 'S23';
tn = 'fenbifactor1_zscore';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_source = 'S23.fenbifactor1';

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
%write to table
T = length(tref);

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
    
    sprintf('%d-%d',i,T)
    
    
end