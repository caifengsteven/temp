%step 1 com factors
%1 day factor
%data  yuqer data
%symbol,tradingdate,f_val
clear
key_str = '合成基于日内指标因子';
print_sel = true;
%parameters
dn = 'S22';
tn = 's22_factor_apb_1d';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_single_day = 's22_factor_apb_1d_single';
tn_single_day_fullname = sprintf('%s.%s',dn,tn_single_day);
%section 1 calculate factor
%check table
%check database

sub_sql = 'show databases';
info = fetchmysql(sub_sql,2);
if ~any(strcmpi(info,dn))
    exemysql(sprintf('create database %s',dn));
end
sub_sql = sprintf('show tables from %s',dn);
info = fetchmysql(sub_sql,2);

%create database ? table ? primary key ? MyISAM
var_info = {'symbol','tradingdate','f_val'};
if ~any(strcmpi(info,tn))
    %create table    
    var_type = cell(size(var_info));
    var_type(:) = {'float'};
    var_type(1:2) = {'varchar(6)','date'};
    obj = mysqlTool();
    sqlquery1=obj.createTable(dn,tn,var_info,var_type);
    OK1 = exemysql(sqlquery1);
    OK2 = exemysql(sprintf('alter table %s.%s engine=MyISAM;',dn,tn));
    OK3 = exemysql(sprintf('alter table %s.%s add primary key(symbol,tradingdate);',dn,tn));
end
%month_cut
tref_end = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tn_single_day_fullname),2);
if ~isempty(tref_end)
    tref_end = tref_end{1};
end
t0 = '2010-01-01';
tref1 = yq_methods.get_tradingdate(t0,tref_end);
tref2 = yq_methods.get_tradingdate_future(tref1{end});
tref = [tref1;tref2(2)];
tref_num = datenum(tref);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
T_month_cut = size(month_cut,1);
month_cut_info = cell(T_month_cut,1);
for i = 1:T_month_cut
    month_cut_info{i} = tref(month_cut(i,1):month_cut(i,2));
end
month_cut_date = tref(month_cut(:,2));
% %symbol
% sql_str = 'select distinct(symbol) from %s order by symbol';
% symbol = fetchmysql(sprintf(sql_str,tn_single_day_fullname),2);
%for symbol do sth
sql_str_f1 = ['select symbol,f_val from %s  where ',...
    ' tradingdate>= ''%s'' and tradingdate <= ''%s''order by tradingdate'];
%write to table
T = size(month_cut,1);

%complete
tref_complete = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tn_fullname),2);

i0 = find(strcmp(month_cut_date,tref_complete))+1;

for i = i0:T
    sub_t_cut = month_cut_info{i};
    syb_t_cut_length = length(sub_t_cut);
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_single_day_fullname,sub_t_cut{1},sub_t_cut{end}),2);
    
    sub_x_symbol = sub_x(:,1);
    symbol = unique(sub_x_symbol);
    sub_x_v = cell2mat(sub_x(:,end));
    
    T_symbol = length(symbol);
    sub_y = nan(T_symbol,1);
    
    parfor j = 1:T_symbol
        sub_sub_y = sub_x_v(strcmp(sub_x_symbol,symbol(j)));
        sub_sub_y(isnan(sub_sub_y)) = [];
        if length(sub_sub_y)<syb_t_cut_length/2
            continue
        end
        sub_y(j) = mean(sub_sub_y);
        if print_sel
            sprintf('%s : %d-%d %d',key_str,j,i,T)
        end
    end
    
    
    sub_symbol_data = [symbol,symbol,num2cell(sub_y)];
    sub_symbol_data(:,2) = month_cut_date(i);
    ia = ~isnan(sub_y);
    sub_symbol_data = sub_symbol_data(ia,:);

    if ~isempty(sub_symbol_data)
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_symbol_data)
        close(conna)
    end
    
    sprintf('%d-%d',i,T)
    
    
end



