%ALTER TABLE wp_comments ENGINE=MYISAM

clear

data_type = {'astock_ycz'};
sel = 1;

db_name = 'futuredata';
tb_name = sprintf('%s_data',data_type{sel});

tb_name_all = sprintf('%s.%s',db_name,tb_name);

var1 = {'symbol','shortname','tradingdate','closeprice'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:2) = {'varchar(8)','date'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);

