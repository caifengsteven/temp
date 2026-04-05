%{
目标将Nincome数据转换为单个季度的
select secID,ticker,publishdate,enddate,endDateRep,actPubtime,updateTime,secShortName,mergedFlag,fiscalPeriod,
reportType,NIncome from yuqerdata.nincome where secID='000001.XSHE' and actPubtime>'2009-01-01' and enddate=endDaterep
order by actPubtime,enddate;

000166开始几年都是按照年度发布报告

update 
加快计算速度
增加同比增速计算
update2
增速计算方法升级
必须上期为正且净利润大于300万才可以为正
%}

clear

%逐个转换并写入数据库
%股票代码，截止日期，报告发布日期，净利润（1个季度）,同比增速，净利润（1个季度）上期,同比增速上期，净利润（1个季度）去年同期
sql_str1 = ['select secID,ticker,publishdate,enddate,fiscalPeriod,',...
        'reportType,NIncome from yuqerdata.nincome where enddate>''2008-01-01'' and enddate=endDaterep ',...
        'and mergedFlag = 1 and secID like ''0%'' or secID like ''60%'' ',... 
        'order by secID,enddate,fiscalPeriod;'];
x = fetchmysql(sql_str1,2);
symbol = unique(x(:,1));
T = length(symbol);

re = cell(11*4*4000,8);
re_ind = 0;


for i = 1:T
    sub_x1=x(strcmp(x(:,1),symbol(i)),:);
    if isempty(sub_x1)
        continue
    end
    %去掉全部以年为频率发布的
    temp = cell2mat(sub_x1(:,5));
    ind = find(eq(temp,3),1);
    if ~isempty(ind)
        sub_x1 = sub_x1(ind:end,:);
    else
        sub_x1 = [];
    end

    if isempty(sub_x1)
        continue
    end
    
    %去掉重复的数据，yuqer数据源相同季度有部分数据重复
    temp = cellfun(@(x,y) [x,'_',y],sub_x1(:,4),sub_x1(:,6),'UniformOutput',false);
    [~,ia] = unique(temp,'stable');
    sub_x1 = sub_x1(ia,:);
        
    num = cell2mat(sub_x1(:,5));
    sub_NIC = cell2mat(sub_x1(:,end));
    ind1 = eq(num,3);
    
    if all(~ind1)
        continue
    end
    
    ind6 = find(eq(num,6)); %半年
    ind9 = find(eq(num,9));
    ind12 = find(eq(num,12));%全年
    
    if ~isempty(ind6)
        sub_NIC6 = sub_NIC(ind6)-sub_NIC(ind6-1);
        sub_re6 = [sub_x1(ind6,[1,4,3]),num2cell(sub_NIC6)];
    else
        sub_re6 = [];
    end
    if ~isempty(ind12)
        sub_NIC12= sub_NIC(ind12)-sub_NIC(ind12-1);
        sub_re12 = [sub_x1(ind12,[1,4,3]),num2cell(sub_NIC12)];
    else
        sub_re12 = [];
    end
    sub_re1 = sub_x1(ind1,[1,4,3,7]);
    
    sub_re = [sub_re1;sub_re6;sub_re12];
    [~,ia] = sort(sub_re(:,2));
    sub_re = sub_re(ia,:);
    %计算同比增速
    sub_t2 = size(sub_re,1);
    if sub_t2>=5
        temp = cell2mat(sub_re(:,end));
        temp1 = nan(sub_t2,1);
        temp1(5:end) = (temp(5:end)-temp(1:end-4))./abs(temp(1:end-4));
        temp_ind = 5:length(temp1);
        temp1(temp_ind(temp(1:end-4)<300*10000)) = 0;
        
        sub_re = [sub_re,num2cell(temp1)];
    else
        sub_re = [sub_re,num2cell(nan(sub_t2,1))];
    end
    %将上期的数据和增速加入数据结构，方便后期检索
    temp1 = cell(sub_t2,3);
    temp1(2:end,1:2) = sub_re(1:end-1,end-1:end);
    temp1(5:end,3) = sub_re(1:end-4,4);
    sub_re = [sub_re,temp1];
    
    sub_t = size(sub_re,1);
    re(re_ind+1:re_ind+sub_t,:) = sub_re;
    re_ind = re_ind+sub_t;
    sprintf('%d-%d',i,T)
end
re = re(1:re_ind,:);
re = [{'symbol','endDate','pubDate','nincome1','nin_rate1','nincome0','nin_rate0','nincome_5'};re];

xlswrite('nincome2.xlsx',re);

