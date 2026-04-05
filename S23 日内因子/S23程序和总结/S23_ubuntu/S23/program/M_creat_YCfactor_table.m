clear
%check database
dN = 'S23';
tn = 'fenbifactor1';
dN_all = fetchmysql('show databases;',2);
if istable(dN_all)
    dN_all = table2cell(dN_all);
end
if ~any(strcmpi(dN_all,dN))
    exemysql(sprintf('create database %s',dN));
end
%check tables `
var_info = {'symbol','tradingdate','spread_date','buy_rate','spread_date_adj'};

tns_all = fetchmysql(sprintf('show tables from %s',dN),2);
if istable(tns_all)
    tns_all = table2cell(tns_all);
end

if ~any(strcmpi(tns_all,tn))
    %create table   
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:2) = {'varchar(6)','date'};
    var_type(7) = {'varchar(6)'};
    obj = mysqlTool();
    sqlquery1=obj.createTable(dN,tn,var_info,var_type);
    OK1 = exemysql(sqlquery1);
    OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dN,tn));
    OK3 = exemysql(sprintf('alter table %s.%s add primary key(symbol,tradingdate);',dN,tn));
end




