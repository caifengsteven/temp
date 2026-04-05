clear

str1 = '%s 数据库安装失败';
str2 = '%s 数据库安装成功';

sql_str = 'show tables from ycz_min_history';
tref = fetchmysql(sql_str,2);

if isempty(tref)
    sprintf(str1,'ycz_min_history')
else
    sprintf(str2,'ycz_min_history')
end


sql_str_bw = 'SELECT ticker,exDivDate,accumAdjFactor FROM yuqerdata.yq_accumadjfactor order by exDivDate desc';
coef_v = fetchmysql(sql_str_bw,2);

if isempty(coef_v)
    sprintf(str1,'yuqerdata')
else
    sprintf(str2,'yuqerdata')
end

sql_str_signal = ['select symbol,closeprice from ycz_result.sta_re20190702_last_30min limit 10 '];
x = fetchmysql(sql_str_signal,2);
if isempty(x)
    sprintf(str1,'ycz_result')
else
    sprintf(str2,'ycz_result')
end
