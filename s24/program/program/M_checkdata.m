clear
%date 
%rate 0  select tradeDate,rate from yuqerdata.shibor_data where ticker = 'Shibor1M' and currency = 'CNY' order by tradeDate;
%
year_t = 244;
ticker1 = fetchmysql('select distinct tickerBond from yuqerdata.convertiblebond_dayprice where tradedate>=''2008-01-01''',2);

ticker2 = fetchmysql('select distinct Liscd from gtadata.BND_Ccbdinfo',2);
ticker2 = cellfun(@num2str,ticker2,'UniformOutput',false);

ticker20 = fetchmysql('select distinct symbol from yuqerdata.bond_impliedvol_wind',2);
ticker20 = cellfun(@num2str,ticker20,'UniformOutput',false);
ticker3 = intersect(ticker1,ticker2);

em_data1 = setdiff(ticker20,ticker3);
em_data2 = setdiff(ticker3,ticker20);

x1 = cell(size(em_data1));
sql1 = 'select shorname from yuqerdata.bond_impliedvol_wind where symbol = ''%s'' limit 1';
for i = 1:length(x1)
    x1(i) = fetchmysql(sprintf(sql1,em_data1{i}),2);
end


x2 = cell(size(em_data2));
sql2 = 'select secShortName from yuqerdata.convertiblebond_info where ticker = ''%s'' limit 1';
for i = 1:length(x2)
    x2(i) = fetchmysql(sprintf(sql2,em_data2{i}),2);
end





