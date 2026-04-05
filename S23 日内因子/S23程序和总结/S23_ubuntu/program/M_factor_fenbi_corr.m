%corr test
clear

tn_f ='S23.fenbifactor1';

tref = fetchmysql(sprintf('select distinct(tradingdate) from %s order by tradingdate',tn_f),2);
if istable(tref)
    tref = table2cell(tref);
end
sql_str = 'select spread_date,buy_rate from S23.fenbifactor1 where tradingdate = ''%s''';

T = length(tref);
r = zeros(T,1);

parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str,tref{i}));
    if istable(sub_x)
        sub_x = table2array(sub_x);
    end
    del_ind = isnan(sum(sub_x,2));
    sub_x(del_ind,:) = [];
    r(i) = corr(sub_x(:,1),sub_x(:,2));
    sprintf('%d-%d',i,T)
end

plot(movmean(r,[20,0]),'LineWidth',2)