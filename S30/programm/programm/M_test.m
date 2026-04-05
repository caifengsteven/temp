clear
sql_str = cell(5,1);
%查看市值月度数据
sql_str{1} = 'select * from S30.mv_month limit 1';
%查看指数成分数据
sql_str{2} = 'select Stkcd from gta_web.gta_idx_smprat limit 1';
%查看yq交易月度数据
sql_str{3} = 'select * from yuqerdata.MktEqumAdjAfGet limit 1';
%查看yq最新数据日期
sql_str{4} = 'select max(tradedate) from yuqerdata.yq_dayprice';
%查看数据立方数据库有多少因子
sql_str{5} = 'show tables from yuqer_cubdata';

for i = 1:length(sql_str)
    i
    disp(fetchmysql(sql_str{i},2))
end
