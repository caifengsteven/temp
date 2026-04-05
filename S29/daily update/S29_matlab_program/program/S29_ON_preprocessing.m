%{
我们首先对各变量取行业内排名，再对
排名进行标准化处理，取各指标对应的 z-score，这样既避免了极端值的
影响，也消除了基本面指标的行业风格特征
s1 date series
s2 get section data
s3 get data industry code
s4 get orders according industry code
s5 zscore
%}

clear

dN= 'S29';
tn = 'factor_yuqer_preprocessing';   
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

%获取时间
sql_str = 'select distinct(pub_date) from S29.factor_yuqer where factor_name = ''%s'' order by pub_date';
sql_str_f2 = ['select symbol,f_val from S29.factor_yuqer where factor_name = ''%s'' and ',...
    'pub_date = ''%s'' and f_val is not null'];
for i = 1:16
    sub_factor_name = sprintf('f%d',i);
    tref = fetchmysql(sprintf(sql_str,sub_factor_name),2);
    T = length(tref);
    parfor j = 1:T
        %get section date
        sub_x = fetchmysql(sprintf(sql_str_f2,sub_factor_name,tref{j}),2);
        %get industry code
        sub_code = yq_methods.get_industry_class(tref{j});
        %arrange data
        [~,ia,ib] = intersect(sub_x(:,1),sub_code(:,1));
        sub_x = sub_x(ia,:);
        sub_x_v = cell2mat(sub_x(:,2));
        sub_code = cell2mat(sub_code(ib,2));
        
        %s4 get orders according industry code
        sub_y = preprocessing_data(sub_code,sub_x_v);
        
        sub_x_update = [sub_x(:,[1,1,1]),num2cell(sub_y)];
        sub_x_update(:,1) = {sub_factor_name};
        sub_x_update(:,2) = tref(j);
        %wrtie to mysql
        if ~isempty(sub_x_update)
            conna = mysql_conn();
            datainsert(conna,tN,var_info,sub_x_update);
            close(conna);            
        end
        sprintf('Complete: %d-%d',i,j)
    end
    
end


