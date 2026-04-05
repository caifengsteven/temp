clear

data_type = {'JJ_future_rehabilitation','YQ_future_rehabilitation'};
sel = 2;

db_name = 'futuredata';
tb_name = sprintf('%s_data',data_type{sel});

tb_name_all = sprintf('%s.%s',db_name,tb_name);
%tradingdate,symbol,close_pirce,open_price,reh_factor
var1 = {'symbol','tradingdate','open_price','close_pirce','reh_factor'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:2) = {'varchar(20)','date'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
