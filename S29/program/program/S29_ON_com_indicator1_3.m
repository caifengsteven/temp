%合成综合指标
%合成前3个指标
clear

%create table
dN = 'S29';
tn = 'factor_yuqer_com';
   
var_info = {'factor_name','pub_date','symbol','f_val'};
var_type = cell(size(var_info));
var_type(:) = {'float'};
var_type(1:3) = {'varchar(6)','date','varchar(6)'};
%key_var = {'symbol','tradingdate'};
key_var = strjoin(var_info([1,2,3]),',');
%key_var = var_info{1};
create_table_adair(dN,tn,var_info,var_type,key_var)

tN = sprintf('%s.%s',dN,tn);
%删除原来数据
sql_str_temp = sprintf('delete from  %s ',tN);
exemysql(sql_str_temp);

factor_contains = {1:4,5:9,10:14};
T_factor_contains = length(factor_contains);
sql_str = 'select distinct(pub_date) from S29.factor_yuqer_preprocessing order by pub_date';
tref = fetchmysql(sql_str,2);
T = length(tref);
sql_str_f1 = 'select factor_name,symbol,f_val from S29.factor_yuqer_preprocessing where pub_date = ''%s''';

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