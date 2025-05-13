%指数数据
%EMA更换为MA
clear
%close all
t0 = datenum(2011,1,4);% start of training of arima
t1 = datenum(2017,1,4);% end of training of arima
t2 = datenum(2017,1,5);% start of test
tt = datenum(2021,12,30); %end of test

index_id = fetchmysql('select distinct index_id from data_pro.main_index_s68',2);
index = table2array(index_id(:,1));
for i=1:length(index)
    index_id_x = '"'+string(index(i))+'"'
    if string(index(i))=='topix'
        c = '"'+"TPX"+'"';
    else
        if string(index(i))=='hsce'
            c = '"'+"HSCEI"+'"';
        else
            c = '"'+string(index(i))+'"';
        end
    end 
    my = 'select * from data_pro.main_index_s68  where index_id ='+index_id_x +' and ticker = '+c+ ' order by tradeDate';
    x =fetchmysql(my,2);


    %x = fetchmysql('select * from data_pro.main_index_s68  where index_id = "RTY" and ticker = "RTY" order by tradeDate',2);
    %x = fetchmysql('select * from data_pro.main_index_s68  where index_id = "hsi" and ticker = "hsi" order by tradeDate',2);
    %x = fetchmysql('select * from data_pro.main_index_s68  where index_id = "topix" and ticker = "TPX" order by tradeDate',2);
    mod = 1;
    %1 全部信号
    %2 只做多
    %3 只做空
    warning('off');

    data_name = {'data000905','data399005'};
    data_info = {'中证500指数','中小板指'};
    vals = [1,1];
    Y = [];
    leg_str = [];
    for sel = 1:2
    sub_data_info = data_info{sel};
    end


    window1 = 22;
    window2 = 22;

    %获取数据
    temp = load(data_name{sel});
    %x =[temp.x1.eob',num2cell([temp.x1.open;temp.x1.close]')];
    %x = [x1.tradeDate, [x1.openPrice; x1.closePrice]'];
    %time
    x.tradeDate = datetime(x.tradeDate,'InputFormat','yyyy-MM-dd');
    %tref_str = datetime(x(:,1),'InputFormat','yyyy-MM-dd');
    %tref = datenum(x(:,1));
    tref =datenum(x.tradeDate);

    ind = tref>=t0&tref<=t1;

    %tref_str = tref_str(ind);
    tref = tref(ind);
    x = x(ind,:);

    %price
    %P_o = cell2mat(x(:,2));
    %P = cell2mat(x(:,5));

    P_o = table2array(x(:,2));
    P_estimate = table2array(x(:,5));

    
    arimaModel = arima(1,1,1);
    Fit = estimate(arimaModel,P_estimate);
    my = 'select * from data_pro.main_index_s68  where index_id ='+index_id_x +' and ticker = '+c+ ' order by tradeDate';
    x =fetchmysql(my,2);
    x.tradeDate = datetime(x.tradeDate,'InputFormat','yyyy-MM-dd');
    tref =datenum(x.tradeDate);
    ind = tref>=t2&tref<=tt;
    tref = tref(ind);
    x = x(ind,:);
    P_o = table2array(x(:,2));
    P = table2array(x(:,5));
    
    %计算指标
    %残差波动率
    %P_ma = MA(P,window1);
    %P_ema = EMA(P,window1);
    number_days = length(P);
    [Y1,YMSE] = forecast(Fit,number_days,P_estimate);
    P_ma = Y1;
    P_ema = Y1;
    T = length(P);
    RVol = zeros(T,1);
    for i = window1:T
        sub_wid = i-window1+1:i;
        %sub_wid = 1:i;
        RVol(i) = std(P(sub_wid)-P_ema(sub_wid))*sqrt(window1);
    end

    RVol_ema = EMA(RVol,window2);
    RVol_std = movstd(RVol-RVol_ema,[window2,0]);
    ind = zeros(T,1);

    for i = max(window1,window2)+1:T-1
        %if RVol(i)-RVol_ema(i)>RVol_std(i)*vals(sel)
        if RVol(i)-RVol_ema(i)>0%RVol_std(i)*vals(sel)
            if P(i)>P_ma(i)
                ind(i) = 1;
                %continue
            elseif P(i)<P_ma(i)
                ind(i) = -1;
                %continue
            else
                ind(i) = 0;
                %continue
            end
        else
            ind(i) = 0;
            %continue
        end
        ind(i+1) = ind(i);    
    end

    r = zeros(size(P));
    r(2:end) = P(2:end)./P(1:end-1)-1;

    ind2 = ind;
    ind2(2:end) =ind(1:end-1);
    ind2(1) = 0;
    if eq(mod,2)
        ind2(ind2<0) = 0;
    elseif eq(mod,3)
        ind2(ind2>0) = 0;
    end
    r1 = r.*ind2;

    Y = cat(2,Y,cumprod(1+[r,r1]));
    leg_str = cat(1,leg_str,{sub_data_info;[sub_data_info,'择时']});

    
    figure;
    plot(tref,Y,'LineWidth',2);
    ah = gca;
    ah.XTickLabelRotation=30;
    datetick('x','yyyymmdd');
    legend(c)
    re = [];
    for i = 1:size(Y,2)
        [v,v_str,sta_val] = curve_static(Y(:,i));
        re = cat(2,re,v');
    end
end