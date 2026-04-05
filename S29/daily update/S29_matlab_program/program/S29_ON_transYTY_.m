%yq因子5、7、9需要转换为同比数值
clear

tN = 'S29.factor_yuqer';
var_info = {'factor_name','pub_date','symbol','f_val'};

tb_id = [5,7,9];

for i = tb_id
    sprintf('%d',i)
    sql_str = 'select symbol,pub_date,f_val from S29.factor_yuqer where factor_name=''-f%d'' order by pub_date,symbol';
    x = fetchmysql(sprintf(sql_str,i),2);
    
    symbol = unique(x(:,1));
    T_symbol = length(symbol);
    F = cell(T_symbol,1);
    parfor j = 1:T_symbol
        ind = strcmp(x(:,1),symbol(j));
        sub_x = x(ind,:);
        sub_t = datevec(sub_x(:,2));
        [~,ia] = sort(sub_t(:,2));
        sub_x = sub_x(ia,:);
        sub_t = sub_t(ia,:);
        sub_x_v = cell2mat(sub_x(:,end));
        
        sub_f = nan(size(sub_x_v));
        for k = 2:size(sub_x,1)
            if eq(sub_t(k,1)-sub_t(k-1,1),1) && eq(sub_t(k,2),sub_t(k-1,2))  
                sub_f(k) = (sub_x_v(k)/sub_x_v(k-1)-1)*100;
            end
            
        end
        del_ind = isnan(sub_f)|isinf(sub_f);
        sub_f = [sub_x(:,1:2),num2cell(sub_f)];
        sub_f(del_ind,:) = [];
        F(j) = {sub_f'};
        sprintf('%d-%d',j,T_symbol)
    end
    F = [F{:}]';
    F = F(:,[1,2,1,3]);
    sub_key = sprintf('f%d',i);
    F(:,1) = {sub_key};
    %insert to database
    
    sql_str_temp = sprintf('delete from  %s where factor_name = ''%s''',tN,sub_key);
    exemysql(sql_str_temp);
    if ~isempty(F)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,F);
        close(conna);            
    end
    
end
