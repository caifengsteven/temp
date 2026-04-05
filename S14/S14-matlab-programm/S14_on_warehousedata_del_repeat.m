%mysql 数据去重
sql_str = 'select * from yuqerdata.yq_MktFutWRdGet where tradeDate >=''2018-01-01''';
x = fetchmysql(sql_str,2);
var = {'tradeDate';'contractObject';'exchangeCD';'unit';'warehouse';'preWrVOL';'wrVOL';'chg'};

temp = cellfun(@(x,y,z) [x,y,z],x(:,1),x(:,2),x(:,5),'UniformOutput',false);
[~,ia] = unique(temp);
x = x(ia,:);
conn= mysql_conn();
datainsert(conn,'yuqerdata.yq_MktFutWRdGet',var,x)
close(conn);