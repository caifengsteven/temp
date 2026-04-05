%{
双因子信号取并集
当收盘价高于结算价，或收盘前半小时委买总量大于委
卖总量时做多，持有至次日上午 10 点平仓。
精细化回测
收益率
%}
clear
signal_str = containers.Map([-1,0,1],{'做空','平仓','做多'});

sql_str_2 = ['SELECT tradeDate,ticker,closeprice-settlePrice FROM ',...
    'yuqerdata.yq_MktMFutdGet where contractObject=''%s'' and mainCon=1 order by tradedate'];

cut_time = [930,1000];
cut_str = {'多头净值','多空净值'};

dns = {'IF','IH','IC'};
T_dns = length(dns);
Y_re = [];

sql_str1 = 'select tradingdate,f_val from S28.comfactors where symbol = ''%s'' and f_type=%d';
sql_str2 = 'select tradingdate,p1,p2,p3 from S28.bac_price where symbol =''%s'' order by tradingdate';

h = figure('Units','normalized','Position',[0,0,1,1],'Name','三因子结果');
movegui(h,'center');

for i0 = 1:T_dns
    sub_str = dns{i0};
    sub_str_l = length(sub_str);
    %x = fetchmysql(sprintf(sql_str,sub_str),2);
    %x = load(sprintf('data_update_%s.mat',sub_str));
    %x = x.F;
    x = fetchmysql(sprintf(sql_str2,sub_str),2);
    
    sub_code = fetchmysql(sprintf(sql_str_2,sub_str),2);
    
    f1 = fetchmysql(sprintf(sql_str1,sub_str,1),2);
    f2 = fetchmysql(sprintf(sql_str1,sub_str,2),2);
    f3 = fetchmysql(sprintf(sql_str1,sub_str,4),2);
    %f1 = load(sprintf('F21_%s.mat',sub_str));
    %f1 = f1.F;
    %f2 = load(sprintf('F22_%s.mat',sub_str));
    %f2 = f2.F;
    
    %f3 = load(sprintf('F24_%s.mat',sub_str));
    %f3 = f3.F;
    sub_inds = suscc_intersect({f1(:,1),f2(:,1),f3(:,1)});
    
    f4 = cell2mat([f1(sub_inds(:,1),2),f2(sub_inds(:,2),2),f3(sub_inds(:,3),2)]);
    
    f4 = sum(f4<0,2);
    f5 = ones(size(f4(:,1)));
    f5(f4>=2) = -1;
    f=[f1(sub_inds(:,1),1),num2cell(f5)];
    
    [~,ia,ib] = intersect(sub_code(:,1),f(:,1),'stable');
    sub_code = [sub_code(ia,1:2),f(ib,end)];
    tref = x(:,1);
    [~,ib,ia] = intersect(tref,sub_code(:,1),'stable');
    tref = tref(ib,:);
    x = x(ib,:);
    %sub_code = sub_code(ia,:);
    
    %if ~eq(length(ia),length(tref))
    %    continue
    %end
    sub_signal = cell2mat(sub_code(ia,3));
    sub_code = sub_code(ia,2);
    sub_code = cellfun(@(x) str2double(x(sub_str_l+1:end)),sub_code);
    sub_code_ind = find(diff(sub_code))+1;

    x = cell2mat(x(:,2:end));
    x = x(:,2)./x(:,3)-1;
    x(isnan(x)|isinf(x)|eq(x,0)) = 0;
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
    y(y<-0.1) = 0;
    y_re = cumsum(y)+1;
    if sub_signal(end)<0
        test_v = 1;
    elseif sub_signal(end)>0
        test_v = -1;
    else
        test_v = 0;
    end
    sub_info = signal_str(test_v);
    
    subplot(3,1,i0)
    bpcure_plot_updateV2(tref,y_re(:,1)*100);
    title(sprintf('%s-%s:%s',tref{end},sub_str,sub_info))
    %setpixelposition(gcf,[223,365,1345,420]);
    box off
    Y_re = cat(2,Y_re,y_re(:,1));

end

%save('final_re_f3.mat','Y_re')