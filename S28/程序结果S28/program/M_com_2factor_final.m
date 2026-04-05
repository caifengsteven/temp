%{
双因子信号取并集
当收盘价高于结算价，或收盘前半小时委买总量大于委
卖总量时做多，持有至次日上午 10 点平仓。
精细化回测
收益率
%}
clear

close all
sql_str_2 = 'SELECT tradeDate,ticker,closeprice-settlePrice FROM futuredata.yuqer_fusharedata where contractObject=''%s'' and mainCon=1 order by tradedate';
cut_time = [930,1000];
cut_str = {'多头净值','多空净值'};

dns = {'IF','IH','IC'};
T_dns = length(dns);
Y_re = [];
for i0 = 1:T_dns
    sub_str = dns{i0};
    sub_str_l = length(sub_str);
    %x = fetchmysql(sprintf(sql_str,sub_str),2);
    x = load(sprintf('data_update_%s.mat',sub_str));
    x = x.F;
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
    tref = x(:,1);
    [~,~,ia] = intersect(tref,sub_code(:,1),'stable');
    if ~eq(length(ia),length(tref))
        continue
    end
    sub_signal = cell2mat(sub_code(ia,3));
    sub_code = sub_code(ia,2);
    sub_code = cellfun(@(x) str2double(x(sub_str_l+1:end)),sub_code);
    sub_code_ind = find(diff(sub_code))+1;

    x = cell2mat(x(:,2:end));
    x = x(:,2)./x(:,3)-1;
    x(isnan(x)) = 0;
    T = length(tref);
    y = zeros(T,2);
    
    y_temp = cell(T,1);
    for i = 2:T
        sub_y = zeros(1,2);
        sub_sub_signal = sub_signal(i-1);       
        sub_x = x(i);
        sub_x = sub_x-2/10000;
        if any(eq(sub_code_ind,i))
            temp=0;
        else
            temp = sub_x;
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
    %y_re = cumprod(1+y);
    y_re = cumsum(y)+1;
    figure
    bpcure_plot_updateV2(tref,y_re(:,1)*100);
    title(sub_str)
    setpixelposition(gcf,[223,365,1345,420]);
    movegui(gcf,'center');
    box off
    Y_re = cat(2,Y_re,y_re(:,1));
end
save('final_re_f1.mat','Y_re')