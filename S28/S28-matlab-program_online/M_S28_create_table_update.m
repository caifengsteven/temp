clear
%增加一张因子表
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S28';
tn = 'comfactors';
   
var_info = {'symbol','tradingdate','f_type','f_val','f_val2'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(8)','date'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = strjoin(var_info(1:3),',');
create_table_adair(dN,tn,var_info,var_type,key_var)


dN= 'S28';
tn = 'bac_price';
   
var_info = {'symbol','tradingdate','p1','p2','p3'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(8)','date'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = strjoin(var_info(1:2),',');
create_table_adair(dN,tn,var_info,var_type,key_var)