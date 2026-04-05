%获取期货上市时间,(这里指在数据库中能检索到的时间)
clear
[~,~,x] = xlsread('future_info.xlsx');
symbol0 = x(:,1);
sy_info0 = x(:,2);

re = cell(length(symbol0),2);

for symbol_sel = 1:length(symbol0)
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');

    sql_str = 'select min(tradingdate),max(tradingdate) from futuredata.price_if_data where variety0=''%s'' and variety=''%s''';
    y_jj = datenum(fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2));
    
    sql_str2 = 'select min(tradingdate),max(tradingdate) from futuredata.future_contracts_data where variety=''%s.%s''';
    sub_sql_str2 = sprintf(sql_str2,symbol{1},symbol{2});
    index_contracts = datenum(fetchmysql(sub_sql_str2,2));
    
    y_jj = [max([y_jj(1),index_contracts(1)]),min([y_jj(2),index_contracts(2)])];
    
    
    
    re(symbol_sel,:) = cellstr(datestr(y_jj,'yyyy-mm-dd'));
    symbol_sel
end