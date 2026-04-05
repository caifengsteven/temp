%经营活动产生的现金流量净额_TTM
clear

tN = 'S30.F4_season';
var_info = {'symbol','pub_date','end_date','f_val'};

%select Stkcd,EndDate,B003000000 from gta_web.stk_fin_incomettm   where B003000000 is not null order by EndDate
sql_str = ['select Stkcd,C001000000 from gta_web.STK_FIN_CashFlowTTM',...
    ' where EndDate = ''%s'' and C001000000 is not null and Stkcd is not null'];

tref_end = ['select distinct(Accper)  from gta_web.IAR_Rept where ',...
    'Accper>=''2000-01-01''  order by Accper'];
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
    sprintf('合成子因子4：Complete: %d-%d',i,T)
    
end