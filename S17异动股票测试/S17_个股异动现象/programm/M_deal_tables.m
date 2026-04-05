%{
更改表格列数据格式
增加主键
%}

%x = fetchmysql('select symbol,close from ycz_min_history.`20130104` order by symbol,tradingdate',2);
t_all = fetchmysql('show tables from ycz_min_history',2);
t_all = cellfun(@str2double,t_all);
t_all = t_all(t_all>=20180101&t_all<=20190101);
T = length(t_all);
parfor i = 2:T
    sub_table = sprintf('ycz_min_history.`%d`',t_all(i));
    sql_str = 'alter table %s modify column symbol varchar(8)';
    exemysql(sprintf(sql_str,sub_table));
    sql_str = 'alter table %s modify column tradingdate datetime';
    exemysql(sprintf(sql_str,sub_table));
    sql_str = 'alter table %s add primary key(symbol,tradingdate)';
    exemysql(sprintf(sql_str,sub_table));
    
    
    %keyboard
    sprintf('%d-%d',i,T)
    
end

