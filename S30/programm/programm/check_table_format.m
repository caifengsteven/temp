function [OK,re] = check_table_format(dN,tn)
    sql_str1 = 'select count(*) from %s.%s';
    sql_str2 = 'desc %s.%s';
    f_c = {'symbol','varchar(6)';'tradingdate','date';'f_val','float'};
    f_c2 = {'symbol','text';'tradingdate','text';'f_val','double'};
    x1 = fetchmysql(sprintf(sql_str1,dN,tn));
    x2 = fetchmysql(sprintf(sql_str2,dN,tn),2);
    if eq(size(x2,1),size(f_c,1)) && size(x2,2)>=2
        if (all(strcmp(x2(:,1),f_c(:,1)) & strcmp(x2(:,2),f_c(:,2)))) || (all(strcmp(x2(:,1),f_c2(:,1)) & strcmp(x2(:,2),f_c2(:,2))))
            x3 = 1;
        else
            x3 = 0;
        end
    end    
    
    if x1 > 5000 && eq(x3,1)
        OK = true;
    else
        OK = false;
    end
    re = {x1,x2};
end