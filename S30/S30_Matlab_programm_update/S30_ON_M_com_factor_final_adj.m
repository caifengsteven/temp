%{
HP滤波
%}

clear
key_str = '价值风格因子HP滤波';
sub_w = 12*5;
tN = 'S30.F_month_final_adj';
var_info = {'symbol','tradingdate','f_val'};

t0 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
month_cut_date = yq_methods.get_month_data();
id = find(strcmp(month_cut_date,t0));
if isempty(id)
    id = sub_w;
end
t0_0 = month_cut_date{id-sub_w+2};
symbol = yq_methods.get_symbol_A();

T = length(symbol);
sql_str = ['select tradingdate,f_val from S30.F_month_final where symbol=''%s''',...
    'and f_val is not null and tradingdate>=''%s'' order by tradingdate'];

X = cell(T,1);
parfor i = 1:T
    sub_symbol = symbol{i};
    sub_x = fetchmysql(sprintf(sql_str,sub_symbol,t0_0),2);
    if size(sub_x,1)<sub_w
        continue
    end
    sub_T = size(sub_x,1);
    sub_x_v = cell2mat(sub_x(:,2));
    sub_y = nan(sub_T,1);
    for j = sub_w:sub_T
        sub_wid = j-sub_w+1:j;
        temp_y = sub_x_v(sub_wid);
        temp = whitsm(temp_y,129600);%趋势项
        sub_y(j) = temp_y(end)-temp(end);
    end
    
    f = [sub_x(:,[1,1]),num2cell(sub_y)];
    f(:,1) = symbol(i);
    f = f(sub_w:end,:);
    X{i} = f';
    sprintf('%s: Complete: %d-%d',key_str,i,T)
end
X = [X{:}]';
%to mysql
if ~isempty(X)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,X);
    close(conna);            
end