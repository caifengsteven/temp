%{
合成APM指标
%}
clear

key_str = 'S32合成STAT指标';
m_start_time = datetime;
print_sel = true;
tN= 'S32.factor_delta';
var_info = {'symbol','tradingdate','f_val'};

tN1 = 'S32.factor_symbolreturn_apm';
tN2 = 'S32.factor_indexreturn_apm';

window1 = 20;
%读取指数数据
sql_str0 = 'select tradingdate,f_am2,f_pm2 from %s where tradingdate>=''%s'' and tradingdate<=''%s'' order by tradingdate';

%x_index = 'select tradingdate,f_am2,f_pm2 from %s order by tradingdate';
%x_index = fetchmysql(sprintf(x_index,tN2),2);

%symbol = fetchmysql(sprintf('select distinct(symbol) from %s',tN1),2);
%sql_str1 = 'select tradingdate,f_am,f_pm from %s where symbol = ''%s'' order by tradingdate';
sql_str1 = 'select symbol,tradingdate,f_am,f_pm from %s where tradingdate>= ''%s'' and tradingdate<=''%s'' order by tradingdate';

%还没有计算的日期
%t1 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
%t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');%从下一个日期开始
%t2 = datestr(now,'yyyy-mm-dd');%当前时间（截至时间）
%tref = yq_methods.get_tradingdate(t1,t2);%没有计算过的时间

tref = yq_methods.get_tradingdate('2020-08-01','2022-08-26');%没有计算过的时间
tref1 = fetchmysql(sprintf('select distinct(tradingdate) from %s where tradingdate>="2020-08-01"',tN),2);
tref = setdiff(tref,tref1);

T = length(tref);
tN1 = 'S32.factor_symbolreturn_apm';
var_info1 = {'symbol','tradingdate','f_am','f_pm'};
for i = 1:T
    
    sql_tmp = 'select * from %s where tradingdate = "%s"';
    sub_x = fetchmysql(sprintf(sql_tmp,tN1,tref{i}),2);
    m = size(sub_x,1);
    if m>5000
        [~,ia] = unique(sub_x(:,1));
        sub_x = sub_x(ia,:);
        exemysql(sprintf('delete from %s where tradingdate = "%s"',tN1,tref{i}));
        conna = mysql_conn();
        datainsert(conna,tN1,var_info1,sub_x);
        close(conna);
        
        
    end
    
end
warning('on')
m_end_time = datetime;
sprintf('Time used %s',m_end_time-m_start_time)