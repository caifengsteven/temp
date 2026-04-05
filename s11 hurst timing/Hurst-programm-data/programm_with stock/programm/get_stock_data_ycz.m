function [index_data,index_name] = get_stock_data_ycz(index_code,t1,t2)
    if nargin < 3
        t1 = '0000-00-01';
    end
    if nargin < 4
        t2='3000-10-10';
    end
    
    if isnumeric(t1)
        t1 = datestr(t1,'yyyy-mm-dd');
    end
    if isnumeric(t2)
        t2 = datestr(t2,'yyyy-mm-dd');
    end
    tn = 'futuredata.astock_ycz_data';
    sql_str = ['select tradingdate,closeprice from %s',10,...
            'where symbol = ''%s'' and tradingdate >=''%s''',10,...
            'and tradingdate <= ''%s'' order by tradingdate'];
    index_data = fetchmysql(sprintf(sql_str,tn,index_code,t1,t2),2);
    
    sql_str = 'select shortname from %s where symbol = ''%s'' limit 1';
    index_name = fetchmysql(sprintf(sql_str,tn,index_code),2);
    if ~isempty(index_name)
        index_name = index_name{1};
    end
end

