%计算分钟级别的IOPV数据
%matlab并行运算时，同一个表格写入可能会有断点写入的操作，会造成假阳性的primary key
%limit 错误， 先写入，最后手动设置primary key即可
clear
tN = 'S31.adj_data';
code_pool = {'510050','510300','510500'};
code_name_pool = {'etf50_min','etf300_min','etf500_min'};
var_info = {'symbol','tradingdate','iopv','openprice','closeprice','volume'};

tref = yq_methods.get_tradingdate('2017-01-13','2020-01-13');
%tref = {'2018-05-02'};
T_tref = length(tref);

for i = 1:3
    code_sel = code_pool{i};
    code_name = code_name_pool{i};
    parfor j = 1:T_tref
        t1 = tref{j};
        %数据格式转换
        t2 = datestr(datenum(t1),'yyyymmdd');
        %载入每日数据
        sql_str = 'select symbol,tradingdate,close from ycz_min_history.`%s`';
        x = fetchmysql(sprintf(sql_str,t2),2);

        %ycz 和 yuqer数据中间转换 symbol 8位 - 6位
        x(:,1) = cellfun(@(x) x(3:end),x(:,1),'UniformOutput',false);
        %卡时间9：30 - 15：00
        sub_t = unique(x(:,2));
        sub_t_num = datenum(sub_t,'yyyy-mm-dd HH:MM:SS');
        sub_t_cut = hour(sub_t_num)*100+minute(sub_t_num);
        sub_t_ind = sub_t_cut>=930 & sub_t_cut<=1501;
        sub_t = sub_t(sub_t_ind);

        %必须现金替代金额
        x1 = fetchmysql(sprintf(['select sum(fixedCahsAmount) from S31.FundETFConsGet where ',...
            'ticker = ''%s'' and tradeDate=''%s'' and cashSubsSign = 2 and fixedCahsAmount is not null'],code_sel,t1));
        if isnan(x1)
            x1 = 0;
        end
        %允许、禁止现金替代成份证券数量
        x2 = fetchmysql(sprintf(['select consTicker,quantity from S31.FundETFConsGet where ',...
            'ticker = ''%s'' and tradeDate=''%s'' and cashSubsSign in(1,3)'],code_sel,t1),2);
        symbol = x2(:,1);
        x2 = cell2mat(x2(:,2));
        %预估现金差额
        x4 = fetchmysql(sprintf(['select estCahComp from S31.FundETFPRListGet where ',...
            'ticker = ''%s'' and tradeDate=''%s'''],code_sel,t1));
        %最小申购赎回单位对应的基金份额
        x5 = fetchmysql(sprintf(['select creationUnit,NAVPreCu from S31.FundETFPRListGet where ',...
            'ticker = ''%s'' and tradeDate=''%s'''],code_sel,t1));

        %最新成交价乘积
        sub_y = nan(length(sub_t),length(symbol));
        for k = 1:length(symbol)
            sub_x = x(strcmp(x(:,1),symbol(k)),2:3);
            [~,ia,ib] = intersect(sub_t,sub_x(:,1),'stable');
            if ~isnan(ia)
                sub_y(ia,k) = cell2mat(sub_x(ib,2));
            end    
        end

        sql_str2 = 'select closeprice from yuqerdata.yq_dayprice where symbol = ''%s'' and tradeDate<=''%s'' order by tradedate desc limit 1;';
        nan_ind = find(isnan(sum(sub_y))); %数据有缺失的，使用过去数据替代
        if ~isempty(nan_ind)
            x3 = nan(size(nan_ind));
            for k = 1:length(nan_ind)
                x3(k) = fetchmysql(sprintf(sql_str2,symbol{nan_ind(k)},t1));
            end
            sub_y(:,nan_ind) = repmat(x3,length(sub_t),1);
        end

        re1 = (x1+sub_y*x2+x4)/x5(1);
        re1 = [sub_t,num2cell(re1)];
        %etf open close volume
        sql_str3 = 'select DATE_ADD(tradingdate,INTERVAL 1 MINUTE),openprice,closeprice,volume from S31.%s where date(tradingdate) = ''%s''';
        re2 = fetchmysql(sprintf(sql_str3,code_name,t1),2);
        
        re1(:,1) = cellfun(@(x) x(1:19),re1(:,1),'UniformOutput',false);
        re2(:,1) = cellfun(@(x) x(1:19),re2(:,1),'UniformOutput',false);
        
        [~,ia,ib] = intersect(re1(:,1),re2(:,1));
        re = [re1(ia,[1,1:end]),re2(ib,2:end)];
        re(:,1) = {code_sel};
        %insert data to mysql
        if ~isempty(re)
            conna = mysql_conn();
            datainsert(conna,tN,var_info,re);
            close(conna);            
        end
        sprintf('Complete : %d-%d %d-%d',i,3,j,T_tref)
    end
end
%添加主键
exemysql('alter table S31.adj_data add primary key(symbol,tradingdate)');