clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S29';
dN2=  'yuqerdata';
tn = 'factor_wind_preprocessing';
   
var_info = {'factor_name','pub_date','symbol','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','varchar(6)'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tn = 'factor_wind_com';
   
var_info = {'factor_name','pub_date','symbol','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','varchar(6)'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tn = 'factor_wind_com_ttm';
   
var_info = {'factor_name','pub_date','symbol','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','varchar(6)'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
tn = 'MktEqumAdjAfGet';
var_info = {'secID', 'ticker', 'secShortName', 'exchangeCD', 'monthBeginDate',...
       'endDate', 'tradeDays', 'preClosePrice', 'openPrice', 'highestPrice',...
       'lowestPrice', 'closePrice', 'turnoverVol', 'turnoverValue', 'chg',...
       'chgPct', 'return', 'turnoverRate', 'avgTurnoverRate', 'varReturn24',...
       'sdReturn24', 'avgReturn24', 'varReturn60', 'sdReturn60',...
       'avgReturn60'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:6) = {'varchar(12)','varchar(6)','varchar(20)','varchar(6)','date','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([2,5]),',');
%key_var = var_info{1};
create_table_adair(dN2,tn,var_info,var_type,key_var)

tn = 'gta_IDX_Smprat';
var_info = {'Indexcd', 'Enddt', 'Stkcd', 'Constdnme', 'Weight'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:4) = {'varchar(12)','date','varchar(12)','varchar(20)'};
key_var = [];
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(var_info([2,5]),',');
%key_var = var_info{1};
create_table_adair('gta_web',tn,var_info,var_type,key_var)