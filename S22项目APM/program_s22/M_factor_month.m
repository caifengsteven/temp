%step 1 com factors
%month day
%data  yuqer data
%symbol,tradingdate,f_val
clear
%parameters
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
t0 = '2010-01-01';
tt = '2019-07-01';
sql_str = 'select distinct tradedate from %s order by tradedate';
tref = fetchmysql(sprintf(sql_str,tn_yq_fullname),2);
tref_num = datenum(tref);
ind_cut = tref_num>=datenum(t0)&tref_num<=datenum(tt);
tref_num = tref_num(ind_cut);
tref = tref(ind_cut);

month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
T_month_cut = size(month_cut,1);
month_cut_info = cell(T_month_cut,1);
for i = 1:T_month_cut
    month_cut_info{i} = tref(month_cut(i,1):month_cut(i,2));
end
month_cut_date = tref(month_cut(:,2));
%symbol
sql_str = 'select distinct(symbol) from %s order by symbol';
symbol = fetchmysql(sprintf(sql_str,tn_yq_fullname),2);
%for symbol do sth
sql_str_f1 = ['select tradedate,turnoverValue/turnovervol*accumAdjFactor ',...
    'as vwap,turnoverVol as vol from %s  where symbol = ''%s'' ',...
    ' and tradedate>= ''',t0,''' and tradedate <= ''',tt,'''order by tradedate'];
%write to table
T = length(symbol);

for i = 1:T
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_yq_fullname,symbol{i}),2);
    if isempty(sub_x)
        continue
    end
    sub_y = nan(T_month_cut,1);
    for j = 1:T_month_cut
        [~,ia] = intersect(sub_x(:,1),month_cut_info{j});
        if ~isempty(ia)
            sub_vwap = cell2mat(sub_x(ia,2));
            sub_vol = cell2mat(sub_x(ia,3));
            %targ = ln(mean(vwap)/(sum(vol*vwap)/vol))
            sub_y(j) = log(mean(sub_vwap)/(sum(sub_vwap.*sub_vol)/sum(sub_vol)));
        end
    end
    
    sub_symbol_data = [month_cut_date,month_cut_date,num2cell(sub_y)];
    sub_symbol_data(:,1) = symbol(i);
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



