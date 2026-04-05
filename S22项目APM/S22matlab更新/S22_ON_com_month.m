%step 1 com factors
%month day
%data  yuqer data
%symbol,tradingdate,f_val
clear
%parameters
key_str = '合成月度APB因子';
print_sel = true;
dn = 'S22';
tn = 's22_factor_apb_month';
tn_fullname = sprintf('%s.%s',dn,tn);

dn_yq = 'yuqerdata';
tn_yq = 'yq_dayprice';
tn_yq_fullname = sprintf('%s.%s',dn_yq,tn_yq);
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
tref_end = fetchmysql(sprintf('select tradedate from %s order by tradedate desc limit 1',tn_yq_fullname),2);
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
%complete data
%complete
tref_complete = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tn_fullname),2);
i0 = find(strcmp(month_cut_date,tref_complete))+1;
%symbol
%for symbol do sth
sql_str_f1 = ['select symbol,tradedate,turnoverValue/turnovervol*accumAdjFactor ',...
    'as vwap,turnoverVol as vol from %s  where ',...
    ' tradedate>= ''%s'' and tradedate <= ''%s''order by tradedate'];
%write to table
T = length(month_cut_date);

for i = i0:T
    sub_t_cut = month_cut_info{i};
    T_sub_t_cut = length(sub_t_cut);
    
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_yq_fullname,sub_t_cut{1},sub_t_cut{end}),2);
    
    symbol = sub_x(strcmp(sub_x(:,2),sub_t_cut(end)),1);
    sub_x_val = cell2mat(sub_x(:,3:4));
    sub_x_symbol = sub_x(:,1);
    
    T_symbol = length(symbol);
    
    sub_y = nan(T_symbol,1);
    parfor j = 1:T_symbol
        ind = strcmp(sub_x_symbol,symbol(j));
        sub_vwap = sub_x_val(ind,1);
        sub_vol = sub_x_val(ind,2);
        sel_ind = ~isnan(sub_vwap+sub_vol);
        sub_vwap = sub_vwap(sel_ind);
        sub_vol = sub_vol(sel_ind);
        if length(sub_vol)<T_sub_t_cut/2
            continue
        end
        
        sub_y(j) = log(mean(sub_vwap)/(sum(sub_vwap.*sub_vol)/sum(sub_vol)));
    end
    
    sub_symbol_data = [symbol,symbol,num2cell(sub_y)];
    sub_symbol_data(:,2) = sub_t_cut(end);
    del_ind = isnan(sub_y);
    sub_symbol_data(del_ind,:) = [];

    if ~isempty(sub_symbol_data)
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_symbol_data)
        close(conna)
    end
    if print_sel
        sprintf('%s: %d-%d',key_str,i,T)
    end
    
    
end



