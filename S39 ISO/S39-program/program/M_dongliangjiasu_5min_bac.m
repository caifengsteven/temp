%分钟框架
clear
key_str = '动量加速分钟回测';
%参数设置
p_window = 60/5*2;
begin_t = '2006-01-01';
end_t = '2020-04-01';
% 股票池的信息日期
info_date = '2015-07-01';
info_date_1year_before = '2014-07-01';
% 回测起始时间
back_date = '2008-06-01';
index_pool = {'000300','000905','000001'};
index_info = {'沪深300','中证500','上证综指'};
for index_sel = 1:length(index_pool)
    index_code = index_pool{index_sel};
    index_name = index_info{index_sel};
    % 收益
    % 月度收益
    sql_str_week_return = ['select endDate,closeprice/precloseprice-1 from yuqerdata.yq_MktEquwAdjAfGet ',...
        'where ticker = "%s" and weekBeginDate>="%s" and tradeDays>0 order by endDate '];

    sql_str_return = ['select date(tradingdate),close from ycz_5min_date.`%d` ',...
        'where symbol = "%s" order by tradingdate '];
    sql_str_cof = 'select tradeDate,accumAdjFactor from yuqerdata.MktEqudAdjAfGet where ticker = ''%s'' order by tradeDate';
    hs_300 = yq_methods.get_index_pool(index_code,info_date);

    if strcmp(index_code,'000001')
        %ST
        st = yq_methods.get_stpt_symbol(info_date);
        hs_300 = setdiff(hs_300,st);
        %B股及次新股
        sql_str =   ['select distinct(ticker)  from yuqerdata.equget  where equTypeCD = "A" ',...
            'and ListSectorCD<=3 and  listDate <="%s" order by ticker'];
        symbol_a = fetchmysql(sprintf(sql_str,info_date_1year_before),2);
        %symbol_a = yq_methods.get_symbol_A();
        hs_300 = intersect(hs_300,symbol_a);
    end
    %月度日期
    tref = yq_methods.get_week_end();
    tref_num = datenum(tref);
    tref_sel = tref_num > datenum(begin_t);
    tref = tref(tref_sel);
    tref_num = tref_num(tref_sel);
    T_tref = length(tref);

    T_symbols = length(hs_300);
    F = cell(T_symbols,1);
    year_num = 2006:2020;
    parfor i = 1:T_symbols
        r_week = fetchmysql(sprintf(sql_str_week_return,hs_300{i},begin_t),2);
        if size(r_week,1)<T_tref*0.4
            F{i} = [zeros(T_tref,1),nan(T_tref,2)];
            continue
        end
        sub_r_week_tref = r_week(:,1);
        r_week = cell2mat(r_week(:,2));
        r_week(isnan(r_week)) = 0;
        [~,ia,ib] = intersect(tref,sub_r_week_tref);
        r_week_re = zeros(T_tref,1);
        r_week_re(ia,:) = r_week(ib);

        %读取股票分钟数据
        sub_symbol = hs_300{i};
        if strcmp(sub_symbol(1),'6')
            sub_symbol1 = ['sh',sub_symbol];
        else
            sub_symbol1 = ['sz',sub_symbol];
        end
        x = cell(size(year_num));
        for j = 1:length(year_num)
            x{j} = fetchmysql(sprintf(sql_str_return,year_num(j),sub_symbol1),2)';
        end
        x=[x{:}]';
        %后复权系数
        sub_tref = x(:,1);
        [sub_tref_u,~,ia] = unique(sub_tref);
        sub_tref_num_u = datenum(sub_tref_u);
        sub_tref_num = sub_tref_num_u(ia);
        x = cell2mat(x(:,2));
        sub_coef = fetchmysql(sprintf(sql_str_cof,hs_300{i}),2);
        sub_coef_tref = sub_coef(:,1);
        sub_coef = cell2mat(sub_coef(:,2));
        sub_sub_coef0 = 1;
        for j = 1:length(sub_tref_u)
            sub_sub_coef = sub_coef(strcmp(sub_coef_tref,sub_tref_u(j)));
            if isempty(sub_sub_coef)
                sub_sub_coef=sub_sub_coef0;
            else
                sub_sub_coef0 = sub_sub_coef;
            end
            sub_sub_ind = eq(ia,j); 
            x(sub_sub_ind) = x(sub_sub_ind)*sub_sub_coef;        
        end
        r=zeros(size(x));
        r(2:end) = x(2:end)./x(1:end-1)-1;
        %每个周内计算系数
        f1 = nan(T_tref,1);
        f2 = f1;
        for j = 2:T_tref
            sub_ind = sub_tref_num>tref_num(j-1) & sub_tref_num<=tref_num(j);
            sub_r = r(sub_ind);
            sub_f1 = S39_methods.get_moment_speed(sub_r,p_window);
            %加速、减速指标
            sub_f2 = S39_methods.get_curve_fit(sub_r,p_window);
            %周内数据求平均
            f1(j) = mean(sub_f1(~isnan(sub_f1)));
            f2(j) = mean(sub_f2(~isnan(sub_f2)));
            sprintf('%s 统计周因子 %d-%d',key_str,j,T_tref)
        end

        F{i} = [r_week_re,f1,f2];
        sprintf('%s %d-%d',key_str,i,T_symbols)
    end

    F = [F{:}];
    %限制回测区间
    tref_sel = tref_num > datenum(back_date);
    tref = tref(tref_sel);
    tref_num = tref_num(tref_sel);
    T_tref = length(tref);
    F = F(tref_sel,:);
    %月度收益及两个指标
    return_df = F(:,1:3:end);
    speed_df = F(:,2:3:end);
    curve_df = F(:,3:3:end);

    %动量上涨、下跌分组
    ind_b_t = S39_methods.get_top_bottom(speed_df);
    r = zeros(T_tref,4);
    for i = 1:T_tref-1
        %加速上涨
        ind1 = ind_b_t{i}{1};
        sub_x = curve_df(i,ind1);
        sub_r = return_df(i+1,ind1);
        sub_r1 = sub_r(sub_x<0);
        sub_r2 = sub_r(sub_x>0);
        if ~isempty(sub_r1)
            r(i+1,1) = mean(sub_r1);    
        end
        if ~isempty(sub_r2)
            r(i+1,2) = mean(sub_r2);    
        end
        %加速下跌
        ind2 = ind_b_t{i}{2};
        sub_x2 = curve_df(i,ind2);
        sub_r = return_df(i+1,ind2);
        sub_r3 = sub_r(sub_x2<0);
        sub_r4 = sub_r(sub_x2>0);
        if ~isempty(sub_r3)
            r(i+1,3) = mean(sub_r3);    
        end
        if ~isempty(sub_r4)
            r(i+1,4) = mean(sub_r4);    
        end
    end
    r(1,:) = 0;
    leg_str = {'top-down','top-up','bot-down','bot-up'};
    S39_methods.draw_figure(tref,r,sprintf('分钟-对冲指数-%s',index_name),leg_str)
    leg_str2 = {'动量上涨加减速','动量下跌加减速'};
    S39_methods.draw_figure(tref,[r(:,1)-r(:,2),r(:,3)-r(:,4)],sprintf('分钟-加减速上涨下跌对冲-%s',index_name),leg_str2)
end
