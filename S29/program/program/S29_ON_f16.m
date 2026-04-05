%流通股总市值 流通股数*月末收盘价
%找到最近的交易日，然后直接读入数据，减少计算量
clear
sprintf('S29:升级因子16')
tN = 'S29.factor_yuqer';
var_info = {'factor_name','pub_date','symbol','f_val'};
 
t0 = 'select pub_date from %s where factor_name = ''f16'' order by pub_date desc limit 1';
t0 = fetchmysql(sprintf(t0,tN),2);

if isempty(t0)
    t0 = '2003-03-01';
else
    t0 = t0{1};
end

m_d = {'03-31','06-30','09-30','12-31'};
year_num = 2003:year(now);

t_cut = zeros(length(year_num)*length(m_d),1);
tref_cut = cell(size(t_cut));
k = 1;
for i = 1:length(year_num)
    for j = 1:length(m_d)
        tref_cut{k} = sprintf('%d-%s',year_num(i),m_d{j});
        sub_t = datenum(sprintf('%d-%s',year_num(i),m_d{j}));
        t_cut(k) = sub_t;
        k = k + 1;
    end
end

tref = yq_methods.get_tradingdate();
tref_num = datenum(tref);

ind = t_cut<=now & t_cut>datenum(t0);
t_cut = t_cut(ind);
tref_cut = tref_cut(ind);

T_tref_cut = length(tref_cut);
sql_str = ['select tradedate,symbol,negMarketValue from yuqerdata.yq_dayprice ',...
        'where tradedate=''%s'' and negMarketValue is not null'];
    
for i = 1:T_tref_cut
    id = find(tref_num<=t_cut,1,'last');
    x = fetchmysql(sprintf(sql_str,tref{id}),2);
    x = x(:,[1,1:end]);
    x(:,1) = {'f16'};
    x(:,2) = tref_cut(i);
    datainsert_adair(tN,var_info,x);    
end

