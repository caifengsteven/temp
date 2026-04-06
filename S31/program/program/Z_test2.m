%计算分钟级别的IOPV数据
%matlab并行运算时，同一个表格写入可能会有断点写入的操作，会造成假阳性的primary key
%limit 错误， 先写入，最后手动设置primary key即可
%补充结果（日期格式不对导致的）
%update 实时数据对接
clear
key_str = '合成IOPV分钟数据';
tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
%code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_name_pool = {'tdx_min_ETF_510050','tdx_min_ETF_510300','tdx_min_ETF_510500'};
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

tref = fetchmysql('show tables from ycz_min_history',2);
tref = cellfun(@(x) [x(1:4),'-',x(5:6),'-',x(7:8)],tref,'UniformOutput',false);
tref_num = datenum(tref);
ind = tref_num>datenum(2018,1,1);
tref = tref(ind);
tref_num = tref_num(ind);

for i = 1
    code_sel = code_pool{i};
    code_name = code_name_pool{i};
    sql_str = 'select distinct(date(tradingdate))  from %s where symbol = ''%s''';
    sub_tref = fetchmysql(sprintf(sql_str,tN,code_sel),2);
    tref1 = setdiff(tref,sub_tref);
    T_tref = length(tref1);
    re_all = cell(T_tref,1);
    for j = 1:T_tref
        t1 = tref1{j};
        %数据格式转换
        t2 = datestr(datenum(t1),'yyyymmdd');
        %载入每日数据
        sql_str = 'select symbol,tradingdate,close from ycz_min_history.`%s`';
        x = fetchmysql(sprintf(sql_str,t2),2);
        if isempty(x)
            sprintf('%s',t2)
        end
    end
end