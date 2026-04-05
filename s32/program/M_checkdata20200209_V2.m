clear

sql_str = 'select distinct(tradingdate) from S32.factor_symbolreturn_apm';
tref1 = fetchmysql(sql_str,2);

sql_str = 'select distinct(tradingdate) from S32.factor_indexreturn_apm';
tref2 = fetchmysql(sql_str,2);

tref3 = yq_methods.get_tradingdate('2013-04-01','2020-01-13');
save checkdata20200209V2