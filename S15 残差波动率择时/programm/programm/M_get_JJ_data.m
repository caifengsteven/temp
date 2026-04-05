addpath('F:\works2019\SOME\项目\gmsdk');

%从掘金下载指数数据
set_token('294931e0418e0fa49482130b72b5ae27e58b0806')
% 获取财务数据（表名，标的名，开始时间，结束时间，表字段）
% x = get_fundamentals('trading_derivative_indicator',{ 'SHSE.600000','SZSE.000001'}, '2018-04-01', '2018-08-01',  {'TCLOSE','NEGOTIABLEMV','TOTMKTCAP','TURNRATE'})


x1 = history({'SHSE.000905'},'1d','2004-12-31','2019-05-01','ADJUST_NONE');
%x1 = history({'SZSE.399005'},'1d','2005-06-07','2019-05-01','ADJUST_NONE');



rmpath('F:\works2019\SOME\项目\gmsdk');


