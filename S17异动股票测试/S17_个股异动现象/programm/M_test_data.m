%{
M_test_data
测试数据库是否安装完成
%}

db_names = {'ycz_min_history.`20130104`','ycz_result.sta_re20190701','ycz_result.sta_re20190702_last_30min',...
    'gtadata.STK_MKT_RepriceFactor'};
info_strs = {'每日分钟数据','统计数据1','收盘30分钟数据','后复权数据'};

T = length(db_names);
sql_str = 'select * from %s limit 10';
sql_str_succ = '%s 数据库安装成功';
sql_str_fail = '%s 数据库安装失败';
for i = 1:T
    sub_sql_str = sprintf(sql_str,db_names{i});
    x = fetchmysql(sub_sql_str,2);
    
    if isempty(x)
        sprintf(sql_str_fail,db_names{i})
    else
        sprintf(sql_str_succ,db_names{i})
    end
    
    
    
end
