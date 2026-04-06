%{
APM指标
构建市场指数
预测者的第一分钟的数据，需要每日数据的收盘数据来计算收益率
只计算月度频率的数据

可以升级为计算日度数据

单个股票 的逐日上午收益率和逐日下午收益率
计算量比较大，需要记录计算时间，以供参考
%}
clear

print_sel = true;
%tN = 'S32.factor_index_min';
%var_info = {'tradingdate','f_val1','f_val2'};
tN1 = 'S32.factor_symbolreturn_apm';
var_info1 = {'symbol','tradingdate','f_am','f_pm'};

tN2 = 'S32.factor_indexreturn_apm';
var_info2 = {'tradingdate','f_am1','f_am2','f_pm1','f_pm2'};

window = 60;
%读取时间
tref = yq_methods.get_tradingdate('2013-04-01','2020-01-13');
tref_num = datenum(tref);
%获取月底日期
%last day for the month
% month_index = month(tref_num);
% month_cut = [0;find(diff(month_index))];
% month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
% month_cut_date1 = tref(month_cut(:,1));
% month_cut_date2 = tref(month_cut(:,2));
% 
% tref = month_cut_date2;
T = size(tref,1);
sql_str1 = 'select * from ycz_min_history.`%s`   limit 1';

re  = zeros(T,1);
parfor i = 2:T
    sub_t = tref{i};
    sub_t = sub_t([1:4,6:7,9:10]);
    x = fetchmysql(sprintf(sql_str1,sub_t),2);
    if isempty(x)
        re(i) = 1;
    end
    sprintf('%d-%d',i,T)
end
save minre re