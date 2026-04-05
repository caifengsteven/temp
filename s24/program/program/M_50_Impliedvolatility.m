clear
%date 
x = fetchmysql(['SELECT tradedate,closeIndex/precloseIndex-1 FROM yuqerdata.yq_index ',...
    'where symbol = ''000016'' and tradedate>=''2008-01-01'' order by tradedate'],2);

y = cell2mat(x(:,2));

y1 = movstd(y,[15,0]);
plot(y1*sqrt(244));

