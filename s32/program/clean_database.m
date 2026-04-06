tN1={'S32.factor_symbolreturn_apm','S32.factor_indexreturn_apm'};

tref=cell(size(tN1));
sql_str='select tradingdate from %s order by tradingdate desc limit 1';
for i = 1:length(tN1)
    tref(i) =fetchmysql(sprintf(sql_str,tN1{i}),2);
end

if ~strcmp(tref(1),tref(2))
    tref=sort(tref);
    tref=tref(1);
    sql_str='delete from %s where tradingdate>''%s''';
    for i =1:2
        exemysql(sprintf(sql_str,tN1{i}));
        
    end
end