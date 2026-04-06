%{
测试文献公式
基金份额参考净值（IOPV）
= (必须现金替代金额
+ 允许、禁止现金替代成份证券数量与最新成交价乘积
+ 预估现金差额)/最小申购赎回单位对应的基金份额
2018-05-24数据缺失
核查，05-23和05-24相同，所以使用05-23代替 
验证
缺失股票处理
%}
clear
tref = yq_methods.get_tradingdate('2017-01-01','2020-01-13');
T_tref = length(tref);
xt = zeros(T_tref,1);
e = nan(T_tref,1);
for i = 403:T_tref
    %try
        t = tref{i};
        x1 = fetchmysql(sprintf(['select sum(fixedCahsAmount) from S31.FundETFConsGet where ',...
            'ticker = ''510050'' and tradeDate=''%s'' and cashSubsSign = 2 and fixedCahsAmount is not null'],t));
        if isnan(x1)
            x1 = 0;
        end

        x2 = fetchmysql(sprintf(['select consTicker,quantity from S31.FundETFConsGet where ',...
            'ticker = ''510050'' and tradeDate=''%s'' and cashSubsSign in(1,3)'],t),2);

        T = size(x2,1);

        sql_str1 ='select symbol,closeprice from yuqerdata.yq_dayprice where tradedate = ''%s''';
        sql_str2 = 'select closeprice from yuqerdata.yq_dayprice where symbol = ''%s'' and tradeDate<=''%s'' order by tradedate desc limit 1;'
        y = fetchmysql(sprintf(sql_str1,t),2);
        [~,ia,ib] = intersect(x2(:,1),y(:,1),'stable');

        x3 = nan(size(x2(:,2)));
        x3(ia) = cell2mat(y(ib,2));
        nan_ind = find(isnan(x3));
        for j = 1:length(nan_ind)
            x3(nan_ind(j)) = fetchmysql(sprintf(sql_str2,x2{nan_ind(j),1},t));
        end


        y = x3;
        x2 = cell2mat(x2(:,2));

        x4 = fetchmysql(sprintf(['select estCahComp from S31.FundETFPRListGet where ',...
            'ticker = ''510050'' and tradeDate=''%s'''],t));
        x5 = fetchmysql(sprintf(['select creationUnit,NAVPreCu from S31.FundETFPRListGet where ',...
            'ticker = ''510050'' and tradeDate=''%s'''],t));


        xt(i) = (x1+sum(x2.*y)+x4)/x5(1);
        sprintf('%d-%d',i,T_tref)
    %catch
        e(i) = 1;
    %end
end