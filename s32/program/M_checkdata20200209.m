clear

sql_str = 'select distinct(tradingdate) from S32.factor_delta';
tref1 = fetchmysql(sql_str,2);

sql_str = 'select distinct(tradingdate) from S32.ret20d_update';
tref2 = fetchmysql(sql_str,2);

sql_str = 'select count(*) from S32.factor_delta';
v1 = fetchmysql(sql_str);

sql_str = 'select count(*) from S32.ret20d_update';
v2 = fetchmysql(sql_str);

save checkdata20200209