%step 1 com factors
%5day month average
%data  yuqer data
%symbol,tradingdate,f_val
clear
key_str = 'S22合成5日因子基础数据';
print_sel = true;
%parameters
window_day = 5;
dn = 'S22';
tn = 's22_factor_apb_5d_single';
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
tref = yq_methods.get_tradingdate('2010-01-01',datestr(now,'yyyy-mm-dd'));
tref_num=datenum(tref);

tref_complete = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tn_fullname),2);
if isempty(tref_complete)
    i0=window_day;
else
    i0=find(strcmp(tref,tref_complete))+1;
end

T = length(tref);
%for symbol do sth
sql_str_f1 = ['select symbol,tradedate,turnoverValue/turnovervol*accumAdjFactor ',...
    'as vwap,turnoverVol as vol from %s  where ',...
    ' tradedate>= ''%s'' and tradedate <= ''%s''order by tradedate'];
%write to table

for i = i0:T
    sub_t_cut = tref(i-window_day+1:i);
       
    sub_x = fetchmysql(sprintf(sql_str_f1,tn_yq_fullname,sub_t_cut{1},sub_t_cut{end}),2);
    if isempty(sub_x)
        continue
    end
    symbol = unique(sub_x(:,1));
    sub_T = length(symbol);
    sub_y0 = nan(sub_T,1);
    parfor j = 1:sub_T
        sub_sub_x = sub_x(strcmp(sub_x(:,1),symbol(j)),:);
        sub_vwap = cell2mat(sub_sub_x(:,3));
        sub_vol = cell2mat(sub_sub_x(:,4));
        %targ = ln(mean(vwap)/(sum(vol*vwap)/vol))
        sub_y0(j) = log(mean(sub_vwap)/(sum(sub_vwap.*sub_vol)/sum(sub_vol)));
        
    end
    
    
    sub_symbol_data = [symbol,symbol,num2cell(sub_y0)];
    sub_symbol_data(:,2) = tref(i);

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



