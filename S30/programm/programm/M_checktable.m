clear

dN = 'yuqer_cubdata';
%dN = 'factors_com';

sql_str = 'show tables from %s';
tns = fetchmysql(sprintf(sql_str,dN),2);

T = length(tns);
sql_str1 = 'select count(*) from %s.%s';
sql_str2 = 'desc %s.%s';

f_c = {'symbol','varchar(6)';'tradingdate','date';'f_val','float'};

y = zeros(T,2);
re = cell(T,1);
for i = 1:T
    x1 = fetchmysql(sprintf(sql_str1,dN,tns{i}));
    x2 = fetchmysql(sprintf(sql_str2,dN,tns{i}),2);
    y(i,1) = x1;
    if all(strcmp(x2(:,1),f_c(:,1)) & strcmp(x2(:,2),f_c(:,2)))
        y(i,2) = 1;
    end
    
    if ~all(y(i,:)>0)
        re{i,1} = tns{i};
        re{i,2} = x2;
    end
    sprintf('%d-%d',i,T)
    
end
save checkre re y