clear
%check database
dN = 'yuqerdata';
tn = 'yq_MktIdxdEvalGet';

var_info = {'secID', 'ticker', 'progFullName', 'secShortName', 'exchangeCD',...
       'tradeDate', 'indexMarketValue', 'indexIncome', 'indexAttrP', 'PEValue',...
       'PEType', 'PB', 'ROE', 'indexValue', 'negIndexValue', 'turnoverRate',...
       'upNum', 'downNum', 'equalNum', 'divYield'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:5) = {'varchar(20)'};
var_type(6) = {'date'};
%key_var = {'symbol','tradingdate'};
key_var = 'ticker,tradeDate,PEType';
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)