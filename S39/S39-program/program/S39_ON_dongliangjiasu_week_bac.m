%新的选股结果写入数据库
%更新一周数据需要19s
clear
key_str = '动量加速周框架数据更新';
%参数设置
p_window = 18*4;
tn = 'S37.symbol_pool_S39';
var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};

%t0_t0 = yq_methods.get_table_end_date(tn,'tradingdate');
sql_str = 'select tradingdate from %s where method_ID=2 order by tradingdate desc limit 1';
t0_t0 = fetchmysql(sprintf(sql_str,tn),2);
t0_t0 = t0_t0{1};
tref = yq_methods.get_week_end();
if strcmp(tref(end),t0_t0)
    sprintf(sprintf('已经是罪行数据，无需 %s',key_str))
    return
end
id = find(strcmp(tref,t0_t0));
% 选股起始时间
back_date = t0_t0;
begin_t = tref{id-p_window*2+1};
% 股票池的信息日期
info_date = t0_t0;
info_date_1year_before = datestr(datenum(info_date)-365,'yyyy-mm-dd');
index_pool = {'000300','000905','000001'};
index_info = {'沪深300','中证500','上证综指'};
symbol_select = cell(size(index_info));
for index_sel = 1:3
    index_code = index_pool{index_sel};
    index_name = index_info{index_sel};
    %指数结果
    sql_str_index = ['select endDate,closePrice from yuqerdata.yq_MktIdxwGet ',...
        'where ticker = ''%s'' and endDate>=''%s'' order by endDate'];
    x_index = fetchmysql(sprintf(sql_str_index,index_code,back_date),2);
    % 月度收益
    sql_str_month_return = ['select endDate,closeprice/precloseprice-1 from yuqerdata.yq_MktEquwAdjAfGet ',...
        'where ticker = "%s" and endDate>="%s" and tradeDays>0 order by endDate '];

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
    tref_sel = tref_num >= datenum(begin_t);
    tref = tref(tref_sel);
    tref_num = tref_num(tref_sel);
    T_tref = length(tref);

    T_symbols = length(hs_300);
    F = cell(T_symbols,1);
    parfor i = 1:T_symbols
        %读取股票数据
        x = fetchmysql(sprintf(sql_str_month_return,hs_300{i},begin_t),2);
        if size(x,1)<p_window
            F{i} = nan(T_tref,3);
            continue
        end
        sub_tref = x(:,1);
        x = cell2mat(x(:,2));
        x(isnan(x)) = 0;
        %动量上涨、下跌指标    
        f1 = S39_methods.get_moment_speed(x,p_window);
        %加速、减速指标
        f2 = S39_methods.get_curve_fit(x,p_window);
        [~,ia,ib] = intersect(tref,sub_tref);
        sub_re = nan(T_tref,3);
        sub_re(:,1) = 0;
        sub_re(ia,:) = [x(ib),f1(ib),f2(ib)];
        %sub_re = [x,f1,f2];
        F{i} = sub_re;
        sprintf('%s %d-%d',key_str,i,T_symbols)
    end
    F = [F{:}];
    %限制回测区间
    tref_sel = tref_num > datenum(back_date);
    tref = tref(tref_sel);
    tref_num = tref_num(tref_sel);
    T_tref = length(tref);
    F = F(tref_sel,:);
    if isempty(F)
        continue
    end
    %月度收益及两个指标
    return_df = F(:,1:3:end);
    speed_df = F(:,2:3:end);
    curve_df = F(:,3:3:end);    
    %动量上涨、下跌分组
    ind_b_t = S39_methods.get_top_bottom(speed_df,30);
    r = zeros(T_tref,4);
    symbol_pool = cell(T_tref,6);
    for i = 1:T_tref
        %加速上涨
        ind1 = ind_b_t{i}{1};
        sub_x = curve_df(i,ind1);
        symbol_pool(i,1) = {hs_300(ind1(sub_x<0))};
        symbol_pool(i,2) = {hs_300(ind1(sub_x>0))};    
        %加速下跌
        ind2 = ind_b_t{i}{2};
        sub_x2 = curve_df(i,ind2);
        symbol_pool(i,3) = {hs_300(ind2(sub_x2<0))};
        symbol_pool(i,4) = {hs_300(ind2(sub_x2>0))};  
        symbol_pool(i,5) = {index_code};
        symbol_pool(i,6) = tref(i);
    end
    symbol_select{index_sel} = symbol_pool';
end
symbol_select = [symbol_select{:}]';
if ~isempty(symbol_select)
    %结果写入数据库,须保留
    %var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};
    T_undo = size(symbol_select,1);
    sub_f1 = cell(T_undo,1);
    sub_f2 = sub_f1;
    for i = 1:T_undo
        sub_code_pool = symbol_select(i,:);
        %more-less more-less
        sub_more1 = strjoin(sub_code_pool{1},',');
        sub_less1 = strjoin(sub_code_pool{2},',');
        sub_more2 = strjoin(sub_code_pool{3},',');
        sub_less2 = strjoin(sub_code_pool{4},',');
        sub_index = sub_code_pool{5};
        sub_t = sub_code_pool{6};
        sub_f1{i} = {sub_t,2,sub_index,sub_more1,sub_less1}';
        sub_f2{i} = {sub_t,3,sub_index,sub_more1,sub_less1}';
    end
    sub_f1 = [sub_f1{:}]';
    sub_f2 = [sub_f2{:}]';

    if ~isempty(sub_f1)
        datainsert_adair(tn,var_info,sub_f1);
    end
    if ~isempty(sub_f2)
        datainsert_adair(tn,var_info,sub_f2);
    end
end
