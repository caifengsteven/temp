%com inverse factor
%合成反转因子
%转换
clear

key_str = 'S32理想反转因子';
m_start_time = datetime;
print_sel = true;

%parameters
dn = 'S32';
tn = 's32_factor_inverse';
tn_fullname = sprintf('%s.%s',dn,tn);

tn_datasource = 'yuqerdata.yq_dayprice';

window_N = 20;
%check table
%check database

sub_sql = 'show databases';
info = fetchmysql(sub_sql,2);
if istable(info)
    info = table2cell(info);
end
if ~any(strcmpi(info,dn))
    exemysql(sprintf('create database %s',dn));
end
sub_sql = sprintf('show tables from %s',dn);
info = fetchmysql(sub_sql,2);
if istable(info)
    info = table2cell(info);
end
%create database ? table ? primary key ? MyISAM
var_info = {'symbol','tradingdate','f_l','f_h','f_val'};
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

%未完成的时间序列
t1 = fetchmysql(sprintf('select tradingdate from %s.%s order by tradingdate desc limit 1',dn,tn),2);
t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');%从下一个日期开始
t2 = datestr(now,'yyyy-mm-dd');%当前时间（截至时间）
tref = yq_methods.get_tradingdate(t1,t2);%没有计算过的时间

% %symbols
% sql_str = 'select distinct(symbol) from %s order by symbol';
% symbol = fetchmysql(sprintf(sql_str,tn_datasource),2);
% if istable(symbol)
%     symbol = table2cell(symbol);
% end
%for symbol do sth
sql_str_f1 = ['select symbol,tradedate,turnovervalue/dealamount,chgPct from %s  where tradeDate >= ''%s'' ',...
    ' and tradeDate<=''%s'' order by tradedate'];
%write to table
T = length(tref);

for i = 1:T
    %时间节点
    %获取交易日期
    sub_tref = yq_methods.get_tradingdate('2000-01-01',tref{i});
    sub_tref = sub_tref(end-window_N+1:end);
    
    X = fetchmysql(sprintf(sql_str_f1,tn_datasource,sub_tref{1},sub_tref{end}),2);
    
    if istable(X)
        X = table2cell(X);
    end
    X_symbol=X(:,1);
    X_tref = X(:,2);
    X_V = cell2mat(X(:,3:end));
    
    symbol = X(strcmp(X_tref,sub_tref(end)),1);
    sub_f = cell(size(symbol));
    T_symbol = length(symbol);
    parfor j = 1:T_symbol
        sub_f{j} = nan(2,1);
        ind = strcmp(X_symbol,symbol(j));
        sub_sub_x = X_V(ind,:);
        if size(sub_sub_x,1)<window_N
            continue
        end
        sub_sub_x(isnan(sum(sub_sub_x,2)),:) = [];
        if size(sub_sub_x,1)<window_N
            continue
        end
        [~,ia] = sort(sub_sub_x(:,1));
        temp1 = cumprod(1+sub_sub_x(ia(1:window_N/2),2));
        temp2 = cumprod(1+sum(sub_sub_x(ia(window_N/2+1:end),2)));
        %sub_f(j,1:2) = [sum(sub_sub_x(ia(1:window_N/2),2)),...
        %    sum(sub_sub_x(ia(window_N/2+1:end),2))];
        %sub_f(j,1:2) = [temp1(end),temp2(end)];
        sub_f{j} = [temp1(end),temp2(end)]';
        
    end
    sub_f = [sub_f{:}]';
    sub_f(:,3) = sub_f(:,2)-sub_f(:,1);
  
    del_ind = isnan(sub_f(:,3));
    sub_f(del_ind,:) = [];
    symbol(del_ind,:) = [];
    if ~isempty(sub_f)
        sub_symbol_data = [symbol,symbol,num2cell(sub_f)];
        sub_symbol_data(:,2) = tref(i);
        conna = mysql_conn();
        %write data to mysql
        datainsert(conna,tn_fullname,var_info,sub_symbol_data)
        close(conna)
    end
    if print_sel
        sprintf('%s: %d-%d',key_str,i,T)
    end
        
end

m_end_time = datetime;
sprintf('Time used %s',m_end_time-m_start_time)