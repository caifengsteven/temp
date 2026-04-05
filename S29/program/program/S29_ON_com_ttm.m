%综合因子4为年度指标，我们使用过去4年数据合成ttm数据，其它指标为季度指标，我们使用
%过去4季度计算
clear

key_str = 'S29 合成TTM';
dN = 'S29';
tn = 'factor_yuqer_com_ttm';
   
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

window = 4;
sql_str_f1 = ['select pub_date,f_val from s29.factor_yuqer_com where ',...
    'factor_name = ''cF%d'' and symbol = ''%s'' order by pub_date'];
sql_str_f2 = 'select distinct(symbol) from s29.factor_yuqer_com where factor_name = ''cF%d''';
sql_str_f3 = 'select count(*) from s29.factor_wind_com where factor_name = ''cF%d''';
T1 = 5;
for i = 1:T1
    symbol = fetchmysql(sprintf(sql_str_f2,i),2);
    T = length(symbol);
    
    re = cell(T,1);
    parfor j = 1:T
        x = fetchmysql(sprintf(sql_str_f1,i,symbol{j}),2);
        
        x_v = cell2mat(x(:,2));
        x_v_ttm = movmean(x_v,[window-1,0]);
        sub_re = [x(:,1),num2cell(x_v_ttm)];
        sub_re = sub_re(window:end,:);
        sub_re = sub_re(:,[1,1,1,2]);
        sub_re(:,1) = {sprintf('ctm%d',i)};
        sub_re(:,3) = symbol(j);
        
        temp = cell2mat(sub_re(:,end));
        del_ind = isnan(temp) | isinf(temp);
        sub_re(del_ind,:) = [];
        re(j) = {sub_re'};
        sprintf('%s: %d-%d %d-%d',key_str,j,T,i,T1)
    end
    re = [re{:}]';
    %wrtie to mysql
    if ~isempty(re)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,re);
        close(conna);            
    end
    
end
