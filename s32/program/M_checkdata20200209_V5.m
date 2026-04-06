%增加中性化处理
%月度
%调仓时，涨停、停牌不买入，跌停、停牌不卖出
%程序不可以在月底时候计算
%APM
clear
print_sel = true;
tN = 'S32.s32_factor_inverse';
tN2 = 'yuqerdata.MktEqumAdjAfGet';
fee = 3/1000;
window = 60;
g_num = 5;
g_str = cell(g_num+1,1);
for i = 1:g_num
    g_str{i} = sprintf('第%d组',i);
end
g_str{end} = '对冲组';

tref = yq_methods.get_tradingdate('2013-05-01','2020-01-13');
tref_num = datenum(tref);
%获取月底日期
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));
tref = month_cut_date2;

tref1 = yq_methods.get_tradingdate(tref{1},'2020-01-13');
[~,ia] = intersect(tref1,tref,'stable');
tref1 = tref1(ia+1);
%tref = tref(datenum(tref)<=datenum(2016,5,31));
T = length(tref);
sql_str1 = 'select symbol,-f_val from %s where tradingdate = ''%s''';
sql_str2 = 'select ticker,closeprice/openprice-1 from %s where enddate = ''%s''';
sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
    'and listDate is not null']; 
sql_str5 = 'select symbol,f_val,log(f_val2) from S32.ret20d where tradingdate = ''%s''';
sql_str6 = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradeDate = ''%s'' and chgPct is not null';
symbol_info = fetchmysql(sql_str4,2);
symbol_listdate = datenum(symbol_info(:,2));
warning('off')
%search symbol
symbol_pool = cell(T,g_num);
symbol_pool_temp = [];
i = 75;%
%因子值
x1 = fetchmysql(sprintf(sql_str1,tN,tref{i}),2);   
%st
st = fetchmysql(sprintf(sql_str3,tref{i}),2);
st = cellfun(@str2double,st,'UniformOutput',false);
st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);

%上市未满 60 日的新股
ind1 = datenum(tref{i})-symbol_listdate>window;

%中性化步骤
x_f = fetchmysql(sprintf(sql_str5,tref{i}),2);
x_indus = yq_methods.get_industry_class_2(tref{i});

tref2 = fetchmysql('select distinct(tradingdate) from S32.s32_factor_inverse',2);

tref3 = fetchmysql('select distinct(tradeDate) from yuqerdata.st_info',2);

tref4 = fetchmysql('select distinct(tradingdate) from S32.ret20d',2);

save checkdata20200209V5.mat