clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dN= 'S28';
tn = 'index300';
   
var_info = {'tradingdate','openprice','highprice','lowprice','closeprice','volume','amt','chgPct'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1) = {'datetime'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%%
tn = 'future300';
var_info = {'markcode', 'ticker', 'tradingdate', 'openprice', 'highprice',...
       'lowprice', 'closeprice', 'volume', 'amt', 'oi', 'chgPct'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(12)','varchar(12)','datetime'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = var_info{3};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%%
tn = 'future300_wind';
var_info = {'markcode', 'ticker', 'tradingdate', 'openprice', 'highprice',...
       'lowprice', 'closeprice', 'volume', 'amt', 'oi', 'chgPct'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(12)','varchar(12)','datetime'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = var_info{3};
create_table_adair(dN,tn,var_info,var_type,key_var)

%%%%%%%%%%%%%%%%%%%%%%%%
tns = {'IF','IH','IC','300','50','500'};
var_info = {'tradingdate', 'openprice', 'highprice', 'lowprice', 'closeprice',...
       'volume', 'amt', 'chg', 'pct_chg', 'oi'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1) = {'datetime'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = var_info{1};
for i = 1:length(tns)
    tn = ['wind_',tns{i}];
    create_table_adair(dN,tn,var_info,var_type,key_var)
end

%%%%%%%%%%%%%%%%%%%%%%
tns = {'IF','IH','IC','300','50','500'};
var_info = {'tradingdate', 'openprice', 'highprice', 'lowprice', 'closeprice',...
       'volume', 'amt', 'chg', 'pct_chg', 'oi'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1) = {'datetime'};
%key_var = {'symbol','tradingdate'};
%key_var = strjoin(key_var,',');
key_var = var_info{1};
for i = 1:length(tns)
    tn = ['wind_',tns{i},'_update'];
    create_table_adair(dN,tn,var_info,var_type,key_var)
end

%%%%%%%%%%%%%%%%%%%%%%
dN2 = 'Future_tick';
var_info = {'marketcode', 'ticker', 'tradingdate', 'infoNow', 'holdInt',...
       'increaseInt', 'turnoverValue', 'turnoverVol', 'openposition',...
       'closeposition', 'type1', 'bsdirection', 'Buy1', 'Buy2', 'Buy3', 'Buy4',...
       'Buy5', 'Sail1', 'Sail2', 'Sail3', 'Sail4', 'Sail5', 'BV1', 'BV2',...
       'BV3', 'BV4', 'BV5', 'SV1', 'SV2', 'SV3', 'SV4', 'SV5'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(12)','varchar(12)','datetime'};
var_type(9:12) = {'varchar(12)'};

%key_var = strjoin(var_info([2,3]),',');
key_var = [];
for y = 2016:2019
    for j = 1:12
        if eq(y,2016) && j<11
            continue
        else
            tn = sprintf('y%d%0.2d',y,j);
            create_table_adair(dN2,tn,var_info,var_type,key_var)
        end
    end
end