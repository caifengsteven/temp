%使用掘金历史数据
%{'沪深300','中证500','创业板指','中小板指','中证1000',...
%        '上证综指','上证50','深次新股','中证流通','深证成指','中证全指'}
function [index_data] = get_index_data_JJ(index_name,t1,t2)
    if nargin < 2
        t1 = '0000-00-01';
    end
    if nargin < 3
        t2='3000-10-10';
    end
    
    if isnumeric(t1)
        t1 = datestr(t1,'yyyy-mm-dd');
    end
    if isnumeric(t2)
        t2 = datestr(t2,'yyyy-mm-dd');
    end
    
    
    tn = 'futuredata.indicator_data';
    sql_str = ['select tradingdate,open,close from %s',10,...
            'where symbolName = ''%s''   and tradingdate >=''%s''',10,...
            'and tradingdate <= ''%s'' order by tradingdate'];
    index_data = fetchmysql(sprintf(sql_str,tn,index_name,t1,t2),2);
    
    
    
    
end

