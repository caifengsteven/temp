clear
%股票周后复权数据
dN= 'yuqerdata';
tn = 'yq_MktEquwAdjAfGet';

var_info = {'secID', 'ticker', 'secShortName', 'exchangeCD', 'weekBeginDate',...
       'endDate', 'tradeDays', 'preClosePrice', 'openPrice', 'highestPrice',...
       'lowestPrice', 'closePrice', 'turnoverVol', 'turnoverValue', 'chg',...
       'chgPct', 'weekreturn', 'turnoverRate', 'avgTurnoverRate', 'varReturn100',...
       'sdReturn100', 'avgReturn100'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:6) = {'varchar(20)','varchar(10)','varchar(10)','varchar(10)','date','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([2,6]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%指数周数据
dN= 'yuqerdata';
tn = 'yq_MktIdxwGet';

var_info = {'indexID', 'ticker', 'secShortName', 'endDate', 'tradeDays',...
       'preClosePrice', 'openPrice', 'highestPrice', 'lowestPrice',...
       'closePrice', 'turnoverVol', 'turnoverValue', 'chg', 'chgPct',...
       'avgPrice', 'wAvgReturn', 'yReturn'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:4) = {'varchar(20)','varchar(10)','varchar(20)' 'date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([2,4]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%弹簧策略选股结果
dN= 'S37';
tn = 'symbol_pool_S39';

var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:5) = {'date','int','varchar(10)','text' 'text'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)
%S39 每日收益统计
%弹簧策略选股结果
dN= 'S37';
tn = 'S39_result';

var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'date','int','varchar(10)'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)
