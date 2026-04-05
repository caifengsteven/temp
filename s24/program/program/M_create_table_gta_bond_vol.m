clear
%check database
dN = 'gtadata';
tn = 'bond_Volatility';
dN_all = fetchmysql('show databases;',2);
if istable(dN_all)
    dN_all = table2cell(dN_all);
end
if ~any(strcmpi(dN_all,dN))
    exemysql(sprintf('create database %s',dN));
end
%check tables `
var_info = {'SecurityID','Symbol','ExchangeCode','ContractCode','ShortName',...
    'TradingDate','RisklessRate','HistoricalVolatility','ImpliedVolatility'};

tns_all = fetchmysql(sprintf('show tables from %s',dN),2);
if istable(tns_all)
    tns_all = table2cell(tns_all);
end

if ~any(strcmpi(tns_all,tn))
    %create table   
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:6) = {'varchar(20)','varchar(20)','varchar(20)','varchar(20)','varchar(20)','date'};
    obj = mysqlTool();
    sqlquery1=obj.createTable(dN,tn,var_info,var_type);
    OK1 = exemysql(sqlquery1);
    OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dN,tn));
    %OK3 = exemysql(sprintf('alter table %s.%s add primary key(tickerBond,tradeDate);',dN,tn));
end
