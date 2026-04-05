%{
合成展期收益率因子
包含以下类型：
第一种是近月和次近月 R1
第二种是近月和主力 R2
第三种是近月和最远月 R3
第四种是主力和次主力 R4
数据库相关
%create table yuqer_fushare_rollreturn
%columns: tradingdate,symbol,R1,R2,R3,R4
update
处理数据为nan现象
由于收盘价很多缺失数据，使用结算价计算。

写入数据库
%}
clear

%create table
var1 = {'tradingdate','symbol','exchangeCD','R1','R2','R3','R4'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:3) = {'date','varchar(10)','varchar(10)'};

db_name = 'futuredata';
tb_name = 'yuqer_future_rollreturn';
obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);

tablename = sprintf('%s.%s',db_name,tb_name);

%获取品种
sql_str_gettype = ['select  distinct contractobject, exchangeCD from futuredata.yuqer_fusharedata',...
    ' where exchangeCD in(''XDCE'',''XSGE'',''XZCE'')'];
fushare_info = fetchmysql(sql_str_gettype,2);
%取第一个品种数据示例


T_f_type = size(fushare_info,1);
conna = database('futuredata','root','liudehua','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/futuredata?useSSL=false&');;
for f_id = 1:T_f_type
    f_code = fushare_info{f_id,1};%期货品种代码
    f_where = fushare_info{f_id,2};%所属交易所
    sub_fn = sprintf('re_%s_%s_update1.csv',f_code,f_where);
    
    [~,~,sub_x] = xlsread(sub_fn);
    sub_x = sub_x(2:end,:);
    %deal isnan value
    for j = 1:4
        temp = sub_x(:,end-j+1);
        temp_ind = cellfun(@isnumeric,temp);
        temp(~temp_ind) = {1e6};
        sub_x(:,end-j+1) = temp;
    end
    
    
    sub_x(:,1) = cellstr(datestr(datenum(sub_x(:,1)),'yyyy-mm-dd'));
    datainsert(conna,tablename,var1,sub_x);
    sprintf('%d',f_id)
    
end

%dos('shutdown -s -t 0')
close(conna)

