clear
%check database
dN = 'S23';
tn = 'zhubifactor_basic';
dN_all = fetchmysql('show databases;',2);
if istable(dN_all)
    dN_all = table2cell(dN_all);
end
if ~any(strcmpi(dN_all,dN))
    exemysql(sprintf('create database %s',dN));
end
%check tables `
var_info = {'symbol','tradingdate','bigB1','bigS1','bigB2','bigS2','focusB',...
    'focusS','Br','Bbeta','Br_adj','Bbeta_adj','BV1','BV2','BV3','BV4','BV5',...
    'BV6','BV7','Sr','Sbeta','Sr_adj','Sbeta_adj','SV1','SV2','SV3','SV4',...
    'SV5','SV6','SV7'};

tns_all = fetchmysql(sprintf('show tables from %s',dN),2);
if istable(tns_all)
    tns_all = table2cell(tns_all);
end

if ~any(strcmpi(tns_all,tn))
    %create table   
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:2) = {'varchar(6)','date'};
    obj = mysqlTool();
    sqlquery1=obj.createTable(dN,tn,var_info,var_type);
    OK1 = exemysql(sqlquery1);
    OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dN,tn));
    OK3 = exemysql(sprintf('alter table %s.%s add primary key(symbol,tradingdate);',dN,tn));
end




