%计算分钟级别的IOPV数据
%matlab并行运算时，同一个表格写入可能会有断点写入的操作，会造成假阳性的primary key
%limit 错误， 先写入，最后手动设置primary key即可
%补充结果（日期格式不对导致的）
%update 实时数据对接
clear
key_str = '合成IOPV分钟数据';
tN = 'S31.adj_data';
code_pool = {'tdx_min_ETF_510050','tdx_min_ETF_510300','tdx_min_ETF_510500'};
%code_name_pool = {'etf50_min','etf300_min','etf500_min'};
code_name_pool = {'tdx_min_ETF_510050','tdx_min_ETF_510300','tdx_min_ETF_510500'};
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

tref = fetchmysql('show tables from ycz_min_history',2);
tref = cellfun(@(x) [x(1:4),'-',x(5:6),'-',x(7:8)],tref,'UniformOutput',false);

tref1 = yq_methods.get_tradingdate('2010-01-01');
setdiff(tref1,tref)


tns = {'yuqerdata.yq_dayprice','yuqerdata.yq_FundETFConsGet','yuqerdata.yq_FundETFPRListGet'};
tref2 = cell(size(tns));
for i = 1:length(tref2)
    tref2{i} = fetchmysql(sprintf('select distinct(tradeDate) from %s order by tradeDate desc limit 10;',tns{i}),2);
end

%pytdx_data
tref3 = cell(size(code_pool));
for i = 1:length(tref3)
    tref3{i} = fetchmysql(sprintf('select distinct(date(tradingdate)) from pytdx_data.%s order by tradingdate desc limit 10;',code_pool{i}),2);
end

save testdata tref2 tref3