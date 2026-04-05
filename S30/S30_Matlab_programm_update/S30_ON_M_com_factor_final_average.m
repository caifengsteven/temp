%{
均线滤波
window = 12*[1,2,3,4,5]
%}

clear
key_str = '价值风格因子移动平均滤波';

tN = 'S30.F_month_final_adj_avg';
var_info = {'symbol','tradingdate','w','f_val'};
t0 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
month_cut_date = yq_methods.get_month_data();
month_cut_date_num = datenum(month_cut_date);
ind = month_cut_date_num>=datenum(t0);
month_cut_date = month_cut_date(ind);
month_cut_date_num = month_cut_date_num(ind);
T_month = length(month_cut_date);
if eq(T_month,1)
    sprintf('%s:Complete!',key_str)
    return
end

symbol = yq_methods.get_symbol_A();

T = length(symbol);
sql_str = 'select tradingdate,f_val from S30.F_month_final where symbol=''%s'' and f_val is not null order by tradingdate';
X = cell(T,1);
parfor i = 1:T
    sub_symbol = symbol{i};
    sub_x = fetchmysql(sprintf(sql_str,sub_symbol),2);
    if isempty(sub_x)
        continue
    end
    sub_T = size(sub_x,1);
    sub_x_v = cell2mat(sub_x(:,2));
    sub_X = cell(5,1);
    for j = 1:5
        sub_w = 12*j;
        sub_y = nan(sub_T,1);
        for k = sub_w:sub_T
            sub_wid = k-sub_w+1:k;
            temp_y = sub_x_v(sub_wid);
            temp = mean(temp_y);%趋势项
            sub_y(k) = temp_y(end)-temp(end);
        end
        f = [sub_x(:,[1,1,1]),num2cell(sub_y)];
        f(:,1) = symbol(i);
        f(:,3) = {j};
        f = f(sub_w:end,:);
        if isempty(f)
            continue
        end
        f = f(datenum(f(:,2))>datenum(t0),:);
        sub_X{j} = f';
        sprintf('%s: Complete: %d-%d,%d',key_str,j,i,T)
    end
    X{i} = [sub_X{:}];
end
X = [X{:}]';
%to mysql
if ~isempty(X)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,X);
    close(conna);            
end


