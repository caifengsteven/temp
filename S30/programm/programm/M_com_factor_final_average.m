%{
均线滤波
window = 12*[1,2,3,4,5]
%}

clear

tN = 'S30.F_month_final_adj_avg';
var_info = {'symbol','tradingdate','w','f_val'};

symbol = fetchmysql('select distinct(symbol) from S30.F_month_final',2);

T = length(symbol);
sql_str = 'select tradingdate,f_val from S30.F_month_final where symbol=''%s'' and f_val is not null order by tradingdate';

parfor i = 1:T
    sub_symbol = symbol{i};
    sub_x = fetchmysql(sprintf(sql_str,sub_symbol),2);
    sub_T = size(sub_x,1);
    sub_x_v = cell2mat(sub_x(:,2));
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

        %to mysql
        if ~isempty(f)
            conna = mysql_conn();
            datainsert(conna,tN,var_info,f);
            close(conna);            
        end
        sprintf('价值风格因子移动平均滤波 Complete: %d-%d,%d',j,i,T)
    end
end
