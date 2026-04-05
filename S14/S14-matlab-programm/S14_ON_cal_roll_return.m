%{
合成展期收益率因子
包含以下类型：
第一种是近月和次近月 R1
第二种是近月和主力 R2
第三种是近月和最远月 R3
第四种是主力和次主力 R4
数据库相关
%create table yuqer_fushare_rollreturn
%columns: tradingdate,symbol,R1,R2,R3,R4
update
处理数据为nan现象
由于收盘价很多缺失数据，使用结算价计算。
update2020/4/11
查看多少时间数据没有计算
逐日计算因子并写入数据库

%}
clear
key_str = 'S14合成展期收益率因子';
table_source = 'yuqerdata.yq_MktMFutdGet';
%create table
var1 = {'tradingdate','symbol','exchangeCD','R1','R2','R3','R4'};
var1_type = cell(size(var1));
var1_type(:) = {'float'};
var1_type(1:3) = {'date','varchar(10)','varchar(10)'};
db_name = 'futuredata';
tb_name = 'yuqer_future_rollreturn';
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

sql_str_delidate = 'select contractobject,ticker,lastdelidate from yuqerdata.yq_FutuGet';
f_detail_date0 = fetchmysql(sql_str_delidate,2);
f_detail_date_num0 = datenum(f_detail_date0(:,3));
sql_str_data1 = ['select contractobject,exchangeCD,ticker,settleprice,contractMark,mainCon,smainCon from %s ',...
        ' where tradedate=''%s'' and exchangeCD in(''XDCE'',''XSGE'',''XZCE'') and ',...
        'settleprice is not null order by ticker'];
    
T_tref = length(tref);
re = cell(T_tref,1);
parfor i = 1:T_tref
    f_detail_date = f_detail_date0;
    f_detail_date_num = f_detail_date_num0;
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
        sub_ind0 = strcmp(f_detail_date(:,1),contractobject_u(j));
        sub_f_detail_date = f_detail_date(sub_ind0,2:end);
        sub_f_detail_date_num = f_detail_date_num(sub_ind0);
        sub_ind = strcmp(contractobject,contractobject_u(j));
        sub_x = x(sub_ind,:);
        sub_r = get_sub_roll_return(sub_x,sub_f_detail_date,sub_f_detail_date_num,datenum(tref(i)));
        sub_r(isnan(sub_r)|isinf(sub_r)) = 1e6;
        
        sub_re{j} = [tref(i),contractobject_u(j),exchangeCD_u(j),num2cell(sub_r)']';
    end
    sub_re = [sub_re{:}];
    re{i} = sub_re;
    sprintf('%s:%d-%d',key_str,i,T_tref)
end
re = [re{:}]';
datainsert_adair(tablename,var1,re)

%dos('shutdown -s -t 0')
%close(conna)

