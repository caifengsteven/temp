%获取S33涉及所有表格的数据的时间节点
%第一，合成中性化需要数据的程序我忘记加了，这个是我的失误，所以到中性化步骤出错
%第二，股指权重数据，我不记得我是否发给过您了
%应该是以上错误导致的最终错误。

clear

tN_all = {'S33.factor_cvar_month','S33.factor_cvar_month_v2'};

tref{1} = fetchmysql(sprintf('select distinct(tradingdate) from %s',tN_all{1}),2);
tref{2} = fetchmysql(sprintf('select distinct(tradingdate) from %s',tN_all{2}),2);

tN3 = 'yuqerdata.MktEqumAdjAfGet';
tref{3} = fetchmysql(sprintf('select distinct(endDate) from %s',tN3),2);

tN4 = 'yuqerdata.st_info';
tref{4} = fetchmysql(sprintf('select distinct(tradeDate) from %s',tN4),2);

sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
                    'and listDate is not null'];
tref{5} = fetchmysql(sql_str4,2);

tN6 = 'yuqerdata.yq_dayprice';
tref{6} = fetchmysql(sprintf('select distinct(tradeDate) from %s',tN6),2);

%中性化所需要的数据表，我忘记了。
tN7 = 'S33.factor_zxh';
tref{7} = fetchmysql(sprintf('select distinct(tradingdate) from %s',tN7),2);

tN8 = 'yuqerdata.IdxCloseWeightGet';
tref{8} = fetchmysql(sprintf('select distinct(tradingdate) from %s',tN8),2);

tN9 = 'yuqerdata.yq_industry_sw';
tref{9} = fetchmysql(sprintf('select count(*) from %s',tN9));

save anal_data20200214 tref