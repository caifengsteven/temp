clear
%check database
dN = 'S26';
tn = 'F1';
   
var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type([1:2]) = {'varchar(6)','date'};
key_var = {'symbol','tradingdate'};
key_var = strjoin(key_var,',');
create_table_adair(dN,tn,var_info,var_type,key_var)

dN = 'S26';
tn = 'S26_result';
   
var_info = {'rule_name','tradingdate','symbol'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'int','date','varchar(6)'};
key_var = strjoin(var_info,',');

create_table_adair(dN,tn,var_info,var_type,key_var)

dN = 'S26';
tn = 'S26_bac';
   
var_info = {'tradingdate','y1','y2','y3'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1) = {'date'};
%key_var = strjoin(var_info,',');
key_var = 'tradingdate';
create_table_adair(dN,tn,var_info,var_type,key_var)