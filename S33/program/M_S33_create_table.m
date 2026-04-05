clear

dN= 'S33';
tn = 'factor_cvar';

var_info = {'symbol','tradingdate','f_val1','f_val2','f_val3','f_val4'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%
dN= 'S33';
tn = 'factor_cvar_v2';

var_info = {'symbol','tradingdate','f_val1','f_val2','f_val3','f_val4'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%
dN= 'S33';
tn = 'factor_cvar_month';

var_info = {'symbol','tradingdate','f_val1','f_val2','f_val3','f_val4'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%
dN= 'S33';
tn = 'factor_cvar_month_v2';

var_info = {'symbol','tradingdate','f_val1','f_val2','f_val3','f_val4'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%
dN= 'S33';
tn = 'factor_zxh';

var_info = {'symbol','tradingdate','f_mv','f_reverse','f_std','f_change'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)