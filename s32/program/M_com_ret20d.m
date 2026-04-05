%{
Ret20d 为前 20 个交易日的涨跌幅
%}
clear
print_sel = true;
tN = 'S32.factor_q';
tN2 = 'yuqerdata.yq_dayprice';
tN3 = 'S32.ret20d';

var_info = {'symbol','tradingdate','f_val','f_val2'};
window = 20;

sql_str = sprintf('select distinct(tradingdate) from %s order by tradingdate',tN);
tref = fetchmysql(sql_str,2);

sql_str = sprintf('select distinct symbol from %s',tN2);
symbol = fetchmysql(sql_str,2);

sql_str1 = ['select tradedate,chgPct,negMarketValue from %s where symbol = ''%s'' ',...
    'and chgPct is not null and negMarketValue is not null order by tradedate'];
T=length(symbol);
parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str1,tN2,symbol{i}),2);
    if isempty(sub_x)
        continue
    end
    sub_x_v = cumprod(1+cell2mat(sub_x(:,2)));
    sub_x_r = zeros(size(sub_x_v));
    sub_x_r(window:end) = sub_x_v(window:end)./sub_x_v(1:end-window+1)-1;
    
    [~,ia,ib] = intersect(tref,sub_x(:,1));
    if isempty(ia)
        continue
    end
    sub_f = [tref(ia),num2cell(sub_x_r(ib)),sub_x(ib,end)];
    sub_f = sub_f(:,[1,1:end]);
    sub_f(:,1) = symbol(i);
    
    temp= sum(cell2mat(sub_f(:,end-1:end)),2);
    sub_f(isnan(temp),:) = [];
    %insert to mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN3,var_info,sub_f);
        close(conna);            
    end
    if print_sel
        sprintf('写入ret20d因子：%d-%d',i,T)
    end
end
