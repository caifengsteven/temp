%{
尾盘涨幅
多头策略：收盘前半小时（15 分钟）下跌做多，持有至次日上午 10 点平仓。
多空策略：收盘前半小时（15 分钟）下跌做多，收盘前半小时（15 分钟）上涨做
空，持有至次日上午 10 点平仓。

Figure
双因子合并做图
%}
clear

signal_str = containers.Map([-1,0,1],{'做空','平仓','做多'});

key_str ='验证23规则';
f_type = 3;
tn_f = 'S28.comfactors';
var_info = {'symbol','tradingdate','f_type','f_val','f_val2'};

sql_str  =['select tradingdate,t_hour*100+t_minute,price from pytdx_data.%s_tdx_min ',...
    'where price is not null and price>0 and tradingdate>=''2016-12-29'' order by tradingdate,t_hour,t_minute'];
sql_str_2 = ['SELECT tradeDate,ticker,closeprice-settlePrice FROM ',...
    'yuqerdata.yq_MktMFutdGet where contractObject=''%s'' and mainCon=1 order by tradedate'];
cut_time = [930,1000];
cut_str = {'多头净值15','多空净值15','多头净值30','多空净值30'};

dns = {'IF','IH','IC'};
T_dns = length(dns);

f0 = fetchmysql(sprintf('select %s from %s where f_type = %d order by tradingdate',...
    strjoin(var_info,','),tn_f,f_type),2);

h = figure('Units','normalized','Position',[0,0,1,1],'Name','尾盘涨幅因子');
movegui(h,'center')
for i0 = 1:T_dns
    
    y1 = [];
    for i1 = 1:2
        sub_str = dns{i0};
        sub_str_l = length(sub_str);
        x = fetchmysql(sprintf(sql_str,sub_str),2);
        sub_code = fetchmysql(sprintf(sql_str_2,sub_str),2);
        f = f0(strcmp(f0(:,1),sub_str),:);
        if eq(i1,1)
            f = f(:,[2,4]);
        else
            f = f(:,[2,4,5]);
        end
        [~,ia,ib] = intersect(sub_code(:,1),f(:,1),'stable');
        sub_code = [sub_code(ia,1:2),f(ib,end)];
        tref_all = x(:,1);
        tref = unique(tref_all);
        [tref,~,ia] = intersect(tref,sub_code(:,1),'stable');
        sub_signal = cell2mat(sub_code(ia,3));
        sub_code = sub_code(ia,2);
        sub_code = cellfun(@(x) str2double(x(sub_str_l+1:end)),sub_code);
        sub_code_ind = find(diff(sub_code))+1;
        t_min = cell2mat(x(:,2));
        x = cell2mat(x(:,3));
        x = [0;x(2:end)./x(1:end-1)-1];

        T = length(tref);
        y = zeros(T,2);

        y_temp = cell(T,1);
        %为了并行，做了调整
        parfor i = 2:T
            sub_y = zeros(1,2);
            sub_sub_signal = sub_signal(i-1);
            sub_ind = strcmp(tref_all,tref(i));        
            sub_x = x(sub_ind);
            sub_t = t_min(sub_ind);

            sub_sub_ind = sub_t>=cut_time(1) &sub_t<=cut_time(2);
            temp0 = sub_x(sub_sub_ind);
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
        y1 = cat(2,y1,y);
    end
    
    y_re = cumprod(1+y1);
    if sub_signal(end)<0
        test_v = 1;
    elseif sub_signal(end)>0
        test_v = -1;
    else
        test_v = 0;
    end
    sub_info = signal_str(test_v);
    
    subplot(3,1,i0)
    T = length(tref);
    plot(y_re*100,'LineWidth',2);
    legend(cut_str,'NumColumns',3,'Location','northwest')
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    t_str = tref(floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str);
    set(gca,'XTickLabelRotation',90)
    title(sprintf('%s-%s:%s',tref{end},sub_str,sub_info))
    %setpixelposition(gcf,[223,365,1345,420]);
    box off
end