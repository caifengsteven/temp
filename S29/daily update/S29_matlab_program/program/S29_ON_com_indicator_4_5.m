%4、5指标为单个指标
%同样写入综合指标数据库便于后续分析
clear
tN = 'S29.factor_yuqer_com';
var_info = {'factor_name','pub_date','symbol','f_val'};
var_info_str = strjoin(var_info,',');
sql_str = 'select %s from S29.factor_yuqer_preprocessing where factor_name=''f%d''';
x = fetchmysql(sprintf(sql_str,var_info_str,15),2);
x(:,1) = {'cF4'};
%wrtie to mysql
if ~isempty(x)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,x);
    close(conna);            
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
x = fetchmysql(sprintf(sql_str,var_info_str,16),2);
x(:,1) = {'cF5'};
%wrtie to mysql
if ~isempty(x)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,x);
    close(conna);            
end
        
