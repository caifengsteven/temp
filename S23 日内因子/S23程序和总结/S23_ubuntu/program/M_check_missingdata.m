clear
%tn_f ='S23.fenbifactor1';
tn_f = 'S23.zhubifactor_basic';

tref1 = fetchmysql(sprintf('select distinct(tradingdate) from %s order by tradingdate',tn_f),2);
if istable(tref1)
    tref1 = table2cell(tref1);
end

tref2 = yq_methods.get_trading_date();
if istable(tref2)
    tref2 = table2cell(tref2);
end

tref2 = tref2(find(strcmp(tref2,tref1(1))):find(strcmp(tref2,tref1(end))));
setdiff(tref2,tref1)