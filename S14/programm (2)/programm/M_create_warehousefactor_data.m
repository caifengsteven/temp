clear

data_type = {'YQ_warehousefactor'};
sel = 1;

db_name = 'futuredata';
tb_name = sprintf('%s_data',data_type{sel});

tb_name_all = sprintf('%s.%s',db_name,tb_name);
%tradingdate,symbol,close_pirce,open_price,reh_factor
var1 = {'tradeDate','contractObject','exchangeCD','wrVOL'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1) = {'date'};
var1_type(2:3) = {'varchar(20)'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
