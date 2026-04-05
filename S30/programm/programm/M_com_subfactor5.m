%+非流动负债合计_最新财报 - 货币资金_最新财报
clear

tN = 'S30.F5_season';
var_info = {'symbol','pub_date','end_date','f_val'};

sql_str = ['select Stkcd,A002200000-A001101000 from gta_web.FS_Combas',...
    ' where Accper = ''%s'' and Stkcd is not null'];

tref_end = ['select distinct(Accper)  from gta_web.IAR_Rept where ',...
    'Accper>=''2000-01-01'' order by Accper'];
tref_end = fetchmysql(tref_end,2);

T = length(tref_end);
for i = 1:T
    %get_date
    x = fetchmysql(sprintf(sql_str,tref_end{i}),2);
    t = gta_web_tool.get_pub_date(tref_end{i});
    if isempty(t) || isempty(x)
        continue
    end
    [~,ia,ib] = intersect(x(:,1),t(:,1));
    X = [t(ib,:),x(ia,:)];
    X(:,3) = tref_end(i);
    nan_ind = cellfun(@isnan,X(:,end));
    X(nan_ind,:) = [];
    %write to mysql
    if ~isempty(X)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,X);
        close(conna);            
    end
    sprintf('合成子因子5：Complete: %d-%d',i,T)
    
end