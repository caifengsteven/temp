%合成综合指标
clear
tN = 'S29.factor_wind_com';
var_info = {'factor_name','pub_date','symbol','f_val'};

factor_contains = {1:4,5:9,10:14};
T_factor_contains = length(factor_contains);
sql_str = 'select distinct(pub_date) from S29.factor_wind order by pub_date';
tref = fetchmysql(sql_str,2);
T = length(tref);
sql_str_f1 = 'select factor_name,symbol,f_val from S29.factor_wind_preprocessing where pub_date = ''%s''';

parfor i = 1:T
    %行业代码
    sub_code = yq_methods.get_industry_class(tref{i});
    x = fetchmysql(sprintf(sql_str_f1,tref{i}),2);
    symbol = unique(x(:,2));
    [symbol,ia] = intersect(sub_code(:,1),symbol);
    indus_code = cell2mat(sub_code(ia,2));
    
    sub_T = length(symbol);
    F = nan(sub_T,T_factor_contains);
    for j = 1:sub_T
        sub_x = x(strcmp(x(:,2),symbol(j)),:);
        sub_f = cellfun(@(x) str2double(x(2:end)),sub_x(:,1));
        for k = 1:T_factor_contains
            [~,~,ia] = intersect(factor_contains{k},sub_f);
            if ~isempty(ia)
                sub_sub_x = cell2mat(sub_x(ia,end));
                F(j,k) = mean(sub_sub_x);
            end
        end
    end
    %preprocessing
    for j = 1:T_factor_contains
        sel_ind = ~isnan(F(:,j));        
        sub_y = preprocessing_data(indus_code(sel_ind),F(sel_ind,j));
        sub_f = [symbol(sel_ind),symbol(sel_ind),symbol(sel_ind),num2cell(sub_y)];
        sub_f(:,1) = {sprintf('cF%d',j)};
        sub_f(:,2) = tref(i);
        %wrtie to mysql
        if ~isempty(sub_f)
            conna = mysql_conn();
            datainsert(conna,tN,var_info,sub_f);
            close(conna);            
        end
        
    end
    sprintf('Complete: %d-%d',i,T)
    
end