%创建统计结果表格
db_name = 'ycz_result';
tb_name = 'sta_re20190702_last_30min';
var1 = {'symbol','tradingdate','precoloseprice','closeprice','r1','d','r2'};
%代码，时间，前收，现收，收益率1，时间间隔，收益率2
var1_type={'varchar(8)','datetime','float','float','float','float','float'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
