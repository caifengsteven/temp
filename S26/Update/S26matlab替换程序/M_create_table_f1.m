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
