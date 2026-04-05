clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S29';
tn = 'factor_wind_preprocessing';
   
var_info = {'factor_name','pub_date','symbol','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','varchar(6)'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

