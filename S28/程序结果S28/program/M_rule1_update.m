%{
F、IH、IC 在隔夜、前收盘至次日上午 10 点、下午开盘后五分钟三个不同时段的
区间累计收益率如下图所示
update
增加了连续合约换约判断，换约当日隔夜收益率不计入
连续合约代码数据来源yuqer

方法流程
1 读入数据后根据不同的时间点，截取各个时间点的收益率，并计算区间收益率；然后计算
收益曲线
2 我们使用 小时×100 + 分钟构建一个指标，用于卡时间
3 回测时如果遇到合约更换，隔夜收益设为0
%}
clear

close all
sql_str  ='select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),pct_chg from S28.wind_%s where pct_chg is not null';
sql_str_2 = 'SELECT tradeDate,ticker FROM futuredata.yuqer_fusharedata where contractObject=''%s'' and mainCon=1 order by tradedate';
cut_time = [930,930;930,1000;1300,1305];
cut_str = {'隔夜','前收盘-10点','下午开盘后五分钟'};

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i0 = 1:T_dns
    sub_str = dns{i0};
    sub_str_l = length(sub_str);
    %日数据数据
    x = fetchmysql(sprintf(sql_str,sub_str),2);
    %合约数据
    sub_code = fetchmysql(sprintf(sql_str_2,sub_str),2);
    %对齐数据
    tref_all = x(:,1);
    tref = unique(tref_all);
    [~,~,ia] = intersect(tref,sub_code(:,1),'stable');
    if ~eq(length(ia),length(tref))
        continue
    end
    sub_code = sub_code(ia,2);
    sub_code = cellfun(@(x) str2double(x(sub_str_l+1:end)),sub_code);
    sub_code_ind = find(diff(sub_code))+1;
    t_min = cell2mat(x(:,2));%小时×100+分钟转换数据
    x = cell2mat(x(:,3));
    
    T = length(tref);
    y = zeros(T,3);

    for i = 1:T
        sub_ind = strcmp(tref_all,tref(i));        
        sub_x = x(sub_ind);
        sub_t = t_min(sub_ind);
        
        for j = 1:3
            sub_sub_ind = sub_t>=cut_time(j,1) &sub_t<=cut_time(j,2); %卡不同的时间段
            temp0 = sub_x(sub_sub_ind);
            if any(eq(sub_code_ind,i)) && j<3
                temp0(1) = 0;
                temp = cumprod(1+temp0)-1;
            else
                temp = cumprod(1+temp0)-1;
            end
            
            y(i,j) = temp(end);%计算时间段内累积收益
        end
        
        sprintf('%d-%d',i,T)
    end
    %y_re = cumprod(1+y)-1;
    y_re = cumsum(y);%合成曲线  单利累加（文献原文）
    figure
    plot(y_re*100,'LineWidth',2);
    legend(cut_str,'NumColumns',3,'Location','northwest')
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    t_str = tref(floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str);
    set(gca,'XTickLabelRotation',90)
    title(sub_str)
    setpixelposition(gcf,[223,365,1345,420]);
    box off

end