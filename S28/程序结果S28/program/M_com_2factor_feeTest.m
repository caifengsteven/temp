%{
双因子信号取并集
当收盘价高于结算价，或收盘前半小时委买总量大于委
卖总量时做多，持有至次日上午 10 点平仓。

收益率
%}
clear

close all
sql_str  =['select date(tradingdate),hour(tradingdate)*100+minute(tradingdate),',...
    'pct_chg from S28.wind_%s where pct_chg is not null and tradingdate>=''2016-12-29''',...
    ' and tradingdate<''2019-11-01'''];
sql_str_2 = ['SELECT tradeDate,ticker,closeprice-settlePrice ',...
    'FROM futuredata.yuqer_fusharedata where contractObject=''%s'' and mainCon=1 order by tradedate'];
cut_time = [930,1000];
cut_str = {'多头净值','多空净值'};
fee_pool = [0,2,3,4,5,8,10]/100/100;
fee_str = num2cell(fee_pool*100);
fee_str = cellfun(@(x) sprintf('%0.2f%%',x),fee_str,'UniformOutput',false);
T_fee = length(fee_pool);

dns = {'IF','IH','IC'};
T_dns = length(dns);
for i0 = 1:T_dns
    
    Y = [];
    for i1 = 1:T_fee
        sub_fee = fee_pool(i1);
        sub_str = dns{i0};
        sub_str_l = length(sub_str);
        x = fetchmysql(sprintf(sql_str,sub_str),2);
        sub_code = fetchmysql(sprintf(sql_str_2,sub_str),2);

        f1 = load(sprintf('F21_%s.mat',sub_str));
        f1 = f1.F;
        f2 = load(sprintf('F22_%s.mat',sub_str));
        f2 = f2.F;
        [~,ia,ib] = intersect(f1(:,1),f2(:,1));
        f3 = cell2mat([f1(ia,2),f2(ib,2)]);
        f4 = ones(size(f3(:,1)));
        f4(f3(:,1)<0|f3(:,2)<0)=-1;
        f=[f1(ia,1),num2cell(f4)];
        [~,ia,ib] = intersect(sub_code(:,1),f(:,1),'stable');
        sub_code = [sub_code(ia,1:2),f(ib,end)];
        tref_all = x(:,1);
        tref = unique(tref_all);
        [~,~,ia] = intersect(tref,sub_code(:,1),'stable');
        if ~eq(length(ia),length(tref))
            continue
        end
        sub_signal = cell2mat(sub_code(ia,3));
        sub_code = sub_code(ia,2);
        sub_code = cellfun(@(x) str2double(x(sub_str_l+1:end)),sub_code);
        sub_code_ind = find(diff(sub_code))+1;
        t_min = cell2mat(x(:,2));
        x = cell2mat(x(:,3));

        T = length(tref);
        y = zeros(T,2);

        y_temp = cell(T,1);
        parfor i = 2:T
            sub_y = zeros(1,2);
            sub_sub_signal = sub_signal(i-1);
            sub_ind = strcmp(tref_all,tref(i));        
            sub_x = x(sub_ind);
            sub_t = t_min(sub_ind);

            sub_sub_ind = sub_t>=cut_time(1) &sub_t<=cut_time(2);
            temp0 = sub_x(sub_sub_ind);
            temp0([1,end]) = temp0([1,end]) - sub_fee/2;
            if any(eq(sub_code_ind,i))
                temp0(1) = 0;
                temp = cumprod(1+temp0)-1;
            else
                temp = cumprod(1+temp0)-1;
            end
            if sub_sub_signal<0
                sub_y(1) = temp(end);
                sub_y(2) = temp(end);
            else
                sub_y(2) = -temp(end);
            end
            y_temp{i} = sub_y;
            sprintf('%d-%d',i,T)
        end
        for i = 2:T
            y(i,:) = y_temp{i};
        end
        Y = cat(2,Y,y(:,1));
    end
    
    %y_re = cumprod(1+Y);
    y_re = 1+cumsum(Y);
    figure
    plot(y_re*100,'LineWidth',2);
    legend(fee_str,'NumColumns',ceil(T_fee/2),'location','best');
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    t_str = tref(floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str);
    set(gca,'XTickLabelRotation',90)
    title(sub_str)
    setpixelposition(gcf,[223,365,1345,420]);
    movegui(gcf,'center');
    box off

end