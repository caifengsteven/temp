%{
为了公平对比指数增强前后的效果，我们使用指数成分股、权重数据合成指数，并在合成指数的
基础上做增强，这样结果更公平
全部使用预测者分钟数据计算
指数增强 使用权市场数据构建指数，权重平均

昨日触发的  昨天信号发出收盘卖出，早盘买入， 当日收益 收盘/开盘-1
今日触发的  今天信号发出收盘卖出，信号收盘/昨收盘-1
两日都未触发的 今收盘/昨天收盘-1
后复权升级

S2022升级
原始程序：M_com_index_ZQ_update_bw_all_com.m

%}

clear

%更新信号
do_n = S17_ON_signal2022();
if do_n
    %回测
    fee1 = 2/10000;
    fee2 = 11/10000;
    index_sel = 1;
    %交易日历获取
    t1 = '2013-01-01';
    t2 = '3033-07-01';

    fn_result ='S2022signal.mat';
    if exist(fn_result,'file')
        signal0 = load(fn_result);
        signal0 = signal0.signal1;
        t0 = datenum( signal0(end,1));
    else
        signal0 = [];
        t0 = datenum('2013-01-01');
    end


    tref = yq_methods.get_tref2(t1,t2);
    tref_num = datenum(tref);
    [tref_num,ia] = sort(tref_num);
    tref = tref(ia);

    %后复权系数
    sql_str_bw = 'SELECT ticker,exDivDate,accumAdjFactor FROM yuqerdata.yq_accumadjfactor order by exDivDate desc';
    coef_v = fetchmysql(sql_str_bw,2);


    sql_str_bw2 = 'select ticker,tradeDate,preClosePrice/actPreClosePrice,accumAdjFactor from yuqerdata.yq_mktequdadjafget where tradeDate = "%s"';


    tref_coeff_num = datenum(coef_v(:,2));
    ind1 = cellfun(@(x) strcmp(x(1),'0'),coef_v(:,1));
    coef_v(ind1,1) = cellfun(@(x) ['sz',x],coef_v(ind1,1),'UniformOutput',false);
    coef_v(~ind1,1) = cellfun(@(x) ['sh',x],coef_v(~ind1,1),'UniformOutput',false);

    %合成
    sql_strb = ['select symbol,open,close from ycz_min_history.`%s` where ',...
        ' time(tradingdate)=''15:00:00'' or time(tradingdate)=''09:31:00'' ',...
        ' order by tradingdate'];

    sql_str_signal = ['select symbol,closeprice from ycz_result.s17_sta_relast_30min ',...
        'where date(tradingdate)=''%s'' and d >=1 '];
    sql_str_signal_2 = ['select symbol,closeprice from ycz_result.s17_sta_relast_30min ',...
        'where date(tradingdate)=''%s'' and d <=-3 '];

    T = length(tref);
    y = zeros(T,1);
    y1 = y;
    parfor i = 2:T
        if tref_num(i)>t0
            sub_x_t0 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i-1),'yyyymmdd')),2);
            sub_x_t0 = arange_yczmin_data(sub_x_t0);%获取开盘、收盘价
            sub_code = sub_x_t0(:,[1,1,2]);
            sub_code(:,end) = num2cell(ones(size(sub_code(:,1)))./size(sub_code,1));
            sub_x_t1 = fetchmysql(sprintf(sql_strb,datestr(tref_num(i),'yyyymmdd')),2);
            sub_x_t1 = arange_yczmin_data(sub_x_t1);

            [inds,commValue] = suscc_intersect({sub_code(:,2),sub_x_t0(:,1),sub_x_t1(:,1)});

            sub_x = [sub_code(inds(:,1),3),sub_x_t0(inds(:,2),2:3),sub_x_t1(inds(:,3),2:3)];
            sub_x = cell2mat(sub_x);
            sub_w_sum = sum(sub_x(:,1));
            sub_x(:,1) = sub_x(:,1)./sub_w_sum;


            tmp = fetchmysql(sprintf(sql_str_bw2,tref{i}),2);
            ind1 = cellfun(@(x) strcmp(x(1),'0'),tmp(:,1));
            tmp(ind1,1) = cellfun(@(x) ['sz',x],tmp(ind1,1),'UniformOutput',false);
            tmp(~ind1,1) = cellfun(@(x) ['sh',x],tmp(~ind1,1),'UniformOutput',false);

            sub_coef0 = tmp(:,[1,2,3]);
            sub_coef1 = tmp(:,[1,2,4]);

            %复权
            [~,ia,ib] = intersect(commValue,sub_coef0(:,1),'stable');
            sub_coeff0_c = ones(size(commValue));
            sub_coeff0_c(ia) = cell2mat(sub_coef0(ib,3));
            sub_x(:,2:3) = sub_x(:,2:3).*repmat(sub_coeff0_c,1,2);

            [~,ia,ib] = intersect(commValue,sub_coef1(:,1),'stable');
            sub_coeff1_c = ones(size(commValue));
            sub_coeff1_c(ia) = cell2mat(sub_coef1(ib,3));
            sub_x(:,4:5) = sub_x(:,4:5).*repmat(sub_coeff1_c,1,2);

            sub_signal_yestoday = fetchmysql(sprintf(sql_str_signal,datestr(tref_num(i-1),'yyyy-mm-dd')),2);
            sub_signal_today = fetchmysql(sprintf(sql_str_signal,datestr(tref_num(i),'yyyy-mm-dd')),2);
            %T+1限制，今日的信号和昨日相同，无法触发（早晨刚买入）
            if  ~isempty(sub_signal_today)&&~isempty(sub_signal_yestoday)
                [~,ia] = setdiff(sub_signal_today(:,1),sub_signal_yestoday(:,1));
                sub_signal_today = sub_signal_today(ia,:);
            end

            temp = (sub_x(:,5)./sub_x(:,3)-1).*sub_x(:,1);
            temp1 = temp;

            if ~isempty(sub_signal_yestoday)
                [~,ia0,ib0] = intersect(sub_signal_yestoday(:,1),commValue);%昨日触发的        
                temp1(ib0) = (sub_x(ib0,5)./(sub_x(ib0,4).*(1+fee1))-1).*sub_x(ib0,1);%昨日触发的
            end
            if ~isempty(sub_signal_today)
                %复权        
                [~,ia1,ib1] = intersect(sub_signal_today(:,1),commValue);%今日触发的
                temp1(ib1) = ((cell2mat(sub_signal_today(ia1,2)).*(1-fee2)).*sub_coeff1_c(ib1)./(sub_x(ib1,3))-1).*sub_x(ib1,1);
            end
            %
            sub_signal_today_2 = fetchmysql(sprintf(sql_str_signal_2,datestr(tref_num(i),'yyyy-mm-dd')),2);
            if ~isempty(sub_signal_today_2)
                %复权        
                [~,ia1,ib1] = intersect(sub_signal_today_2(:,1),commValue);%今日触发的
                temp1(ib1) = temp1(ib1)+(sub_x(ib1,5)*(1-fee2)./(cell2mat(sub_signal_today_2(ia1,2)).*(1+fee1).*sub_coeff1_c(ib1))-1).*sub_x(ib1,1);
            end

            %temp = (sub_x(:,3)./sub_x(:,2)-1).*sub_x(:,1)./sum(sub_x(:,1));
            y(i) = sum(temp);
            y1(i) = sum(temp1);
            sprintf('%d-%d',i,T)
        end
    end

    signal1 = [tref,num2cell([y,y1])];
    if ~isempty(signal0)
        ind = tref_num>t0;
        signal1 = [signal0;signal1(ind,:)];
    end
    save(fn_result,'signal1');

    y = cell2mat(signal1(:,2));
    y1 = cell2mat(signal1(:,3));
    y(isinf(y)) = 0;
    y1(isnan(y1)) = 0;

    index_code = {'000001','000300','000905'};
    Y = zeros(length(tref),3);
    for i = 1:length(index_code)
        sql_tmp = 'select tradeDate,CHGPct from yuqerdata.yq_index where symbol = "%s" and tradeDate>="%s"';
        sub_x = fetchmysql(sprintf(sql_tmp,index_code{i},tref{1}),2);
        [~,ia,ib] = intersect(tref,sub_x(:,1));
        Y(ia,i) = cell2mat(sub_x(ib,2));
    end

    %t_str = index_code;
    t_str1 = ['合成指数',index_code];
    t_str2 = cellfun(@(x) ['S17对冲-',x],index_code,'UniformOutput',false);
    Y1=[y1,Y];
    Y2 = repmat(y1,1,size(Y,2)) - Y;

    y_re1 = cumprod(1+Y1);
    y_re2 = cumprod(1+Y2);
    H(1) = bacFigure(y_re1,tref,[],t_str1);
    H(2) = bacFigure(y_re2,tref,[],t_str2);
    Y = [y_re1,y_re2];

    report_adair('S17计算结果',H,Y,[t_str1,t_str2]);
    A_S17_ON_signal;
end


