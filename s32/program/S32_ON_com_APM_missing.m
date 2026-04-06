%合成APM因子
%升级程序，终版本

clear
key_str = '合成APM因子';
m_start_time = datetime;

print_sel = true;
tN= 'S32.factor_apm';
var_info = {'symbol','tradingdate','f_val'};
%{
t1 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');%从下一个日期开始
t2 = datestr(now,'yyyy-mm-dd');%当前时间（截至时间）
tref = yq_methods.get_tradingdate(t1,t2);%没有计算过的时间
%}
tref = yq_methods.get_tradingdate('2020-08-01','2022-08-26');%没有计算过的时间
tref1 = fetchmysql('select distinct(tradingdate) from S32.factor_apm where tradingdate>="2020-08-01"',2);
tref = setdiff(tref,tref1);

tref_num = datenum(tref);

sql_str1 = 'select symbol,f_val from S32.factor_delta where tradingdate = ''%s''';
sql_str2 = 'select symbol,f_val from S32.ret20d_update where tradingdate = ''%s''';

T = length(tref);
r = zeros(T,1);
r2 = r;
for i = 1:T
    warning('off')
    
    x = fetchmysql(sprintf(sql_str2,tref{i}),2); %ret20
    y = fetchmysql(sprintf(sql_str1,tref{i}),2); %delta

    inds = suscc_intersect({y(:,1),x(:,1)});
    sub_symbol = x(inds(:,2),1);
    
    y = cell2mat(y(inds(:,1),2));
    x = cell2mat(x(inds(:,2),2));    
    
    [~,~,resi] = regress(y,x);
    
    sub_f = [sub_symbol,sub_symbol,num2cell(resi)];
    sub_f(:,2) = tref(i);
    
    %into mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);
    end
    
    if print_sel
        sprintf('%s:%d-%d',key_str,i,T)
    end
end

warning('on')
m_end_time = datetime;
sprintf('Time used %s',m_end_time-m_start_time)
