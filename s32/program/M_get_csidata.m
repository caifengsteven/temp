%获取股票池每日数据
clear
%股票池选择
symbol_pool_all = {'000300','000905','000985','000852'};
symbol_pool_info = {'300股票池','500股票池','中证全指股票池','中证1000股票池'};
sel_id = 1;
symbol_pool=symbol_pool_all{sel_id};
symbol_pool_info = symbol_pool_info{sel_id};
fn = sprintf('csi%s.csv',symbol_pool);
%读取交易日期

tref = yq_methods.get_tradingdate('2013-04-01','2020-01-13');
tref_num = datenum(tref);
%获取每日数据
T = length(tref);
sql_str1 = ['select symbol,tradedate,openprice,lowestprice,',...
    'highestprice,closeprice from yuqerdata.yq_dayprice where tradedate = ''%s'''];
sql_str2 = 'select ticker,accumAdjFactor from yuqerdata.MktEqudAdjAfGet where tradeDate = ''%s''';
X = cell(T,1);
for i = 1:T
    sub_t = fetchmysql(sprintf(...
        'select tradingdate from s22.index_com where tradingdate <= ''%s'' and ticker = ''%s'' order by tradingdate desc limit 1',...
        tref{i},symbol_pool),2);
    if isempty(sub_t)
        continue
    end
    sub_symbol_pool = fetchmysql(sprintf(...
        'select symbol from s22.index_com where tradingdate = ''%s'' and ticker = ''%s''',sub_t{1},symbol_pool),2);
    
    sub_x = fetchmysql(sprintf(sql_str1,tref{i}),2);
    sub_y = fetchmysql(sprintf(sql_str2,tref{i}),2);
    
    %后复权
    [~,ia,ib] = intersect(sub_x(:,1),sub_y(:,1));
    sub_x_v = cell2mat(sub_x(ia,3:end));
    sub_y_v = cell2mat(sub_y(ib,2));
    
    sub_x_v = bsxfun(@times,sub_x_v,sub_y_v);
    sub_x = [sub_x(ia,1:2),num2cell(sub_x_v)];
    
    [~,ia] = intersect(sub_x(:,1),sub_symbol_pool);
    sub_x = sub_x(ia,:);
    X{i} = sub_x';
    
    sprintf('获取%s %d-%d',symbol_pool_info,i,T)
    
end
X = [X{:}]';

X = cell2table(X,'VariableNames',{'symbol','tradedate','openprice','lowestprice','highestprice','closeprice'});

writetable(X,fn);
