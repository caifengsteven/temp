clear

tns = cell(5,1);
for i = 1:5
    tns{i} = sprintf('F%d_season',i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S30';
 
var_info = {'symbol','pub_date','end_date','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};

for i = 1:5
    tn = tns{i};
    create_table_adair(dN,tn,var_info,var_type,key_var)
end

%%%%%%%%%%%%%%%%%%%%%%%%

tns = cell(5,1);
for i = 1:5
    tns{i} = sprintf('F%d_month',i);
end
tns = cat(1,tns,{'F_month_final';'F_month_final_adj'});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S30'; 
var_info = {'symbol','tradingdate','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:2) = {'varchar(6)','date'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2]),',');
%key_var = var_info{1};

for i = 1:length(tns)
    tn = tns{i};
    create_table_adair(dN,tn,var_info,var_type,key_var)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dN= 'S30';
tn = 'F_month_final_adj_avg';

var_info = {'symbol','tradingdate','w','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','int'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

