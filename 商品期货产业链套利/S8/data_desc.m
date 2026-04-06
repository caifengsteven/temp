clear
%{
sql_str = 'select tradingdate,volume from futuredata.price_shfe_data where codename in (''RB1801'',''RB1901'') and year(tradingdate) = 2018 order by tradingdate;';
x1 = fetchmysql(sql_str,2);

sql_str = 'select tradingdate,volume from futuredata.price_shfe_data where codename in (''RB1805'',''RB1905'') and year(tradingdate) = 2018 order by tradingdate;';
x5 = fetchmysql(sql_str,2);

sql_str = 'select tradingdate,volume from futuredata.price_shfe_data where codename in (''RB1809'',''RB1909'') and year(tradingdate) = 2018 order by tradingdate;';
x9 = fetchmysql(sql_str,2);
%}
tref =fetchmysql( 'select distinct tradingdate from futuredata.price_shfe_data where variety = ''RB''',2);
tref_num = datenum(tref);
ind = tref_num>=datenum(2014,7,28)&tref_num<=datenum(2016,12,31);
tref = tref(ind);
tref_num = tref_num(ind);

T = length(tref);
X = cell(T,1);
parfor i = 1:T
    sql_str = 'select codename,volume from futuredata.price_shfe_data where tradingdate = ''%s'' and variety = ''RB'' order by volume desc limit 1';
    X{i} = fetchmysql(sprintf(sql_str,tref{i}),2);
end


x1=cellfun(@(x) x{1},X,'UniformOutput',false);
x2=cellfun(@(x) x{2},X,'UniformOutput',false);
v1 = datevec(tref);
v2 = (v1(:,1)-2000)*100+v1(:,2);

y = cellfun(@(x) str2double(x(end-1:end)),x1);

t = datetime(tref_num,'ConvertFrom','datenum');
bar(t,y);
