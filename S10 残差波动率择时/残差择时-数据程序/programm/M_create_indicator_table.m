clear

data_type = {'indicator'};
sel = 1;

db_name = 'futuredata';
tb_name = sprintf('%s_data',data_type{sel});

tb_name_all = sprintf('%s.%s',db_name,tb_name);

var1 = {'tradingdate';'symbols';'symbolName';'open';'close';'high';'low';'amount';'volume'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:3) = {'date','varchar(12)','varchar(12)'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1,'ycz_zhubi','root','');
sqlquery2 = obj.addKey(db_name,tb_name,'key1',var1(1:3));
OK2 = exemysql(sqlquery2,'ycz_zhubi','root','');