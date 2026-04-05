%{
查看分钟数据是否缺失
%}
clear

print_sel = true;

%读取时间
tref = yq_methods.get_tradingdate('2013-04-01','2020-01-13');
tref_num = datenum(tref);
T = size(tref,1);
sql_str1 = 'select * from ycz_min_history.`%s`   limit 1';

re  = zeros(T,1);
parfor i = 1:T
    sub_t = tref{i};
    sub_t = sub_t([1:4,6:7,9:10]);
    x = fetchmysql(sprintf(sql_str1,sub_t),2);
    if isempty(x)
        re(i) = 1;
    end
    sprintf('%d-%d',i,T)
end

ind = eq(re,1);
if any(ind)
    sprintf('以下分钟数据缺失%s',strjoin(tref(ind),','))
else
    
    sprintf('分钟数据完好')
    
end