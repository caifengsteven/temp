%{
计算期货单日所有合约总成交量
update 2020/4/11
对接yuqer数据，对接程序

%}
clear
key_str = 'S14合成基差动量因子';
table_source = 'yuqerdata.yq_MktMFutdGet';
%create table
var1 = {'tradingdate','symbol','exchangeCD','R1'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:3) = {'date','varchar(10)','varchar(10)'};
db_name = 'futuredata';
tb_name = 'yuqer_totalvolume';
obj = mysqlTool();
sqlquery1=obj.createTable(db_name,tb_name,var1,var1_type);
OK1 = exemysql(sqlquery1);
tablename = sprintf('%s.%s',db_name,tb_name);

t0 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tablename),2);
tt = fetchmysql(sprintf('select tradeDate from %s order by tradeDate desc limit 1',table_source),2);
tref = yq_methods.get_tradingdate(t0{1},tt{1});
tref = tref(2:end);
if isempty(tref)
    sprintf('%s已经是最新的',key_str)
    return
end

% sql_str_delidate = 'select contractobject,ticker,lastdelidate from yuqerdata.yq_FutuGet';
% f_detail_date0 = fetchmysql(sql_str_delidate,2);
% f_detail_date_num0 = datenum(f_detail_date0(:,3));
sql_str_data1 = ['select contractobject,exchangeCD,ticker,openInt,contractMark,mainCon,smainCon from %s ',...
        ' where tradedate=''%s'' and exchangeCD in(''XDCE'',''XSGE'',''XZCE'') and ',...
        'settleprice is not null order by ticker'];
    
T_tref = length(tref);
re = cell(T_tref,1);
parfor i = 1:T_tref
    x = fetchmysql(sprintf(sql_str_data1,table_source,tref{i}),2);
    contractobject = x(:,1);
    exchangeCD = x(:,2);
    x = x(:,3:end);
    [~,ia] = unique(contractobject);
    contractobject_u = contractobject(ia);
    exchangeCD_u = exchangeCD(ia);
    T_contractobject = length(contractobject_u);
    sub_re = cell(T_contractobject,1);
    for j = 1:T_contractobject
        sub_ind = strcmp(contractobject,contractobject_u(j));
        sub_x = x(sub_ind,:);
        sub_r = sum(cell2mat(sub_x(:,2)));
        sub_r(isnan(sub_r)|isinf(sub_r)) = 1e6;
        
        sub_re{j} = [tref(i),contractobject_u(j),exchangeCD_u(j),num2cell(sub_r)]';
    end
    sub_re = [sub_re{:}];
    re{i} = sub_re;
    sprintf('%s:%d-%d',key_str,i,T_tref)
end
re = [re{:}]';
datainsert_adair(tablename,var1,re)

%dos('shutdown -s -t 0')
%close(conna)

