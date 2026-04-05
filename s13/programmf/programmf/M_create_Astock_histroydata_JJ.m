%ALTER TABLE wp_comments ENGINE=MYISAM

clear

data_type = {'JJAstockdata_ADJUST_NONE'};
sel = 1;

db_name = 'futuredata';
tb_name = data_type{sel};

tb_name_all = sprintf('%s.%s',db_name,tb_name);

var1 = {'tradingdate','symbol','openprice','closeprice','lowprice','highprice','volume','amount'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:2) = {'date','varchar(11)'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
sqlquery2 = obj.addKey(db_name,tb_name,'key1',var1(1:2));
OK2 = exemysql(sqlquery2);

exemysql(sprintf('ALTER TABLE %s.%s ENGINE=MYISAM',db_name,tb_name));