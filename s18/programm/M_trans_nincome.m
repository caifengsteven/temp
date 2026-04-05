%{
目标将Nincome数据转换为单个季度的
select secID,ticker,publishdate,enddate,endDateRep,actPubtime,updateTime,secShortName,mergedFlag,fiscalPeriod,
reportType,NIncome from yuqerdata.nincome where secID='000001.XSHE' and actPubtime>'2009-01-01' and enddate=endDaterep
order by actPubtime,enddate;

000166开始几年都是按照年度发布报告
%}

clear
%获取symbol
sql_str = ['select distinct secID from yuqerdata.nincome where (secID like ''0%'' ',...
    'or secID like ''6%'') and secShortName not like ''PT%'' order by secID'];
symbol = fetchmysql(sql_str,2);

T = length(symbol);
%逐个转换并写入数据库
%股票代码，截止日期，报告发布日期，净利润（1个季度）
sql_str1 = ['select secID,ticker,publishdate,enddate,fiscalPeriod,',...
        'reportType,NIncome from yuqerdata.nincome where secID=''%s'' and enddate>''2009-01-01'' and enddate=endDaterep ',...
        'and mergedFlag = 1 order by enddate,fiscalPeriod;'];
re = cell(11*4*4000,4);
re_ind = 0;


for i = 303:T
    x=fetchmysql(sprintf(sql_str1,symbol{i}),2);
    if isempty(x)
        continue
    end
    %去掉全部以年为频率发布的
    temp = cell2mat(x(:,5));
    ind = find(eq(temp,3),1);
    if ~isempty(ind)
        x = x(ind:end,:);
    else
        x = [];
    end

    if isempty(x)
        continue
    end
    
    %去掉重复的数据，yuqer数据源相同季度有部分数据重复
    temp = cellfun(@(x,y) [x,'_',y],x(:,4),x(:,6),'UniformOutput',false);
    [~,ia] = unique(temp,'stable');
    x = x(ia,:);
        
    num = cell2mat(x(:,5));
    sub_NIC = cell2mat(x(:,end));
    ind1 = eq(num,3);
    
    if all(~ind1)
        continue
    end
    
    ind6 = find(eq(num,6)); %半年
    ind9 = find(eq(num,9));
    ind12 = find(eq(num,12));%全年
    
    if ~isempty(ind6)
        sub_NIC6 = sub_NIC(ind6)-sub_NIC(ind6-1);
        sub_re6 = [x(ind6,[1,4,3]),num2cell(sub_NIC6)];
    else
        sub_re6 = [];
    end
    if ~isempty(ind12)
        sub_NIC12= sub_NIC(ind12)-sub_NIC(ind12-1);
        sub_re12 = [x(ind12,[1,4,3]),num2cell(sub_NIC12)];
    else
        sub_re12 = [];
    end
    sub_re1 = x(ind1,[1,4,3,7]);
    
    sub_re = [sub_re1;sub_re6;sub_re12];
    [~,ia] = sort(sub_re(:,2));
    sub_re = sub_re(ia,:);
    
    sub_t = size(sub_re,1);
    re(re_ind+1:re_ind+sub_t,:) = sub_re;
    re_ind = re_ind+sub_t;
    sprintf('%d-%d',i,T)
end

re = re(1:re_ind,:);




