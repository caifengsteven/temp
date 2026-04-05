%创建统计结果表格
db_name = 'juejindata';
tb_name = 'backward_data';
var1 = {'symbol','tradingdate','pre_close','open','high','low','close','volume','amount'};
%代码，时间，前收，现收，收益率1，时间间隔，收益率2
var1_type=cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:2)={'varchar(12)','date'};

obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
