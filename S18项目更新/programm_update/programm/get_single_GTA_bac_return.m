function sub_y  = get_single_GTA_bac_return(sub_symbol,sub_t,sub_tref)
    fee1 = 1/1000;
    fee2 = 2/1000;
    sql_str_m1 = ['select symbol,date(tradingdate),closeprice/precloseprice-1 from futuredata.STK_MKT_BWARDQUOTATION where symbol in(%s) ',...
    'and tradingdate>=''%s'' and tradingdate<=''%s'' and filling = 0 order by tradingdate;'];
    
    sub_str1 = sprintf('''%s''',strjoin(sub_symbol,''','''));
    sub_str2 = datestr(sub_t(1),'yyyy-mm-dd');
    sub_str3 = datestr(sub_t(2),'yyyy-mm-dd');
    
    sub_x = fetchmysql(sprintf(sql_str_m1,sub_str1,sub_str2,sub_str3),2);
    
    sub_y = zeros(length(sub_tref),length(sub_symbol));
    for j = 1:length(sub_symbol)
        sub_sub_x = sub_x(strcmp(sub_x(:,1),sub_symbol(j)),:);
        [~,ia] = intersect(sub_tref,sub_sub_x(:,2),'stable');
        sub_y(ia,j) = cell2mat(sub_sub_x(:,3));
    end
    sub_y(1,:) = sub_y(1,:)-fee1;%ÂòÈë
    sub_y(end,:) = sub_y(end,:) - fee2;%Âô³ö
end