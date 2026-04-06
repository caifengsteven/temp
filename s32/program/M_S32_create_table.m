clear

dN= 'S32';
tn = 'factor_q';

var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S32';
tn = 'ret20d';

var_info = {'symbol','tradingdate','f_val','f_val2'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)
%%%%%%%%%%%%%%%%%%%
dN= 'S32';
tn = 'ret20d_update';

var_info = {'symbol','tradingdate','f_val','f_val2'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%% factor_index_min
dN= 'S32';
tn = 'factor_index_min';

var_info = {'tradingdate','f_val1','f_val2'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'datetime','float'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(var_info([1,2]),',');
key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%% factor_apm
dN= 'S32';
tn = 'factor_apm';

var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%
dN= 'S32';
tn = 'factor_symbolreturn_apm';

var_info = {'symbol','tradingdate','f_am','f_pm'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%
dN= 'S32';
tn = 'factor_indexreturn_apm';

var_info = {'tradingdate','f_am1','f_am2','f_pm1','f_pm2'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1) = {'date'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(var_info([1,2]),',');
key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)
%%%%%%%%%%%%
dN= 'S32';
tn = 'factor_delta';

var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%
dN= 'S32';
tn = 'factor_delta0';

var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%
dN= 'S32';
tn = 'rankIC_data';

var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%
dN= 'S32';
tn = 'com_factor';

var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)