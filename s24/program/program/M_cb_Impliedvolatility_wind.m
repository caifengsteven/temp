clear
%date 
tref = fetchmysql('SELECT distinct(tradingdate) FROM yuqerdata.bond_impliedvol_wind where tradingdate >=''2008-01-01'' order by tradingdate',2);

T = length(tref);
sql_str = 'select f_val from yuqerdata.bond_impliedvol_wind where tradingdate =''%s''';
y = zeros(T,1);
parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str,tref{i}));
    sub_x(isnan(sub_x)) = [];
    y(i) = mean(sub_x);
end

y_ci = movmean(y,[15,0])/100;

x = fetchmysql(['SELECT tradedate,closeIndex/precloseIndex-1 FROM yuqerdata.yq_index ',...
    'where symbol = ''000016'' and tradedate>=''2008-01-01'' order by tradedate'],2);

y2 = cell2mat(x(:,2));
y1 = movstd(y2,[15,0]);

y_50 = movmean(y1*sqrt(244),[15,0]);
[~,ia,ib] = intersect(tref,x(:,1));
tref = tref(ia);
Y = [y_ci(ia),y_50(ib)];
Y_m = Y(:,2)./Y(:,1);
