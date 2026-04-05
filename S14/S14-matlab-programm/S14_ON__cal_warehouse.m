%{
update
处理数据为nan现象
call warehouse factor 2020/4/11 update

%}
clear
key_str = '更新S14仓单因子';
table_source = 'yuqerdata.yq_MktFutWRdGet';
%create table
var1 = {'tradeDate','contractObject','exchangeCD','wrVOL'};
db_name = 'futuredata';
tb_name = 'yq_warehousefactor_data'; %factorname
tablename = sprintf('%s.%s',db_name,tb_name);

t0 = fetchmysql(sprintf('select tradeDate from %s order by tradeDate desc limit 1',tablename),2);
tt = fetchmysql(sprintf('select tradeDate from %s order by tradeDate desc limit 1',table_source),2);

tref = yq_methods.get_tradingdate(t0{1},tt{1});
tref = tref(2:end);
if isempty(tref)
    sprintf('%s已经是最新的',key_str)
    return
end
T_tref = length(tref);
sql_str1 = ['select contractObject,exchangeCD,wrvol from %s where tradeDate = ''%s'' ',...
    'and exchangeCD in(''XDCE'',''XSGE'',''XZCE'') and wrvol is not null'];
re = cell(T_tref,1);
parfor i = 1:T_tref
    x = fetchmysql(sprintf(sql_str1,table_source,tref{i}),2);
    sub_code = unique(x(:,1));
    T_sub_code = length(sub_code);
    sub_re = cell(T_sub_code,1);
    for j = 1:T_sub_code
        ia = find(strcmp(x(:,1),sub_code(j)));
        sub_x = cell2mat(x(ia,end));
        temp = [tref(i),x(ia(1),1:2),num2cell(sum(sub_x))];
        sub_re{j} = temp';
    end
    re{i} = [sub_re{:}];
    sprintf('%s complete:%d-%d',key_str,i,T_tref)
end
re = [re{:}]';
datainsert_adair(tablename,var1,re)
%dos('shutdown -s -t 0')
%close(conna)

