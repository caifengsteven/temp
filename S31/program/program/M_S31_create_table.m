clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S31';
tn = 'adj_data';
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','datetime'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(var_info([1,2]),',');
key_var = [];
create_table_adair(dN,tn,var_info,var_type,key_var)


