%创建统计结果表格
db_name = 'joinQuantData';
tb_name = 'astock_index';
var1 = {'symbol','shortname','tradingdate','weight'};
%代码，时间，前收，现收，收益率1，时间间隔，收益率2
var1_type={'varchar(12)','varchar(20)','date','float'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
