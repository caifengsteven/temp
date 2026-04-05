clear
key_str = '动量加速周框架';
%参数设置
p_window = 18*4;
begin_t = '2006-01-01';
end_t = '2020-04-01';
tn = 'S37.symbol_pool_S39';
var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};

% 股票池的信息日期
info_date = '2015-07-01';
info_date_1year_before = '2014-07-01';
% 回测起始时间
back_date = '2008-06-01';
index_pool = {'a','000300','000905','000001'};
index_info = {'全市场','沪深300','中证500','上证综指'};
for index_sel = 1:4
    index_code = index_pool{index_sel};
    index_name = index_info{index_sel};
    %指数结果
    sql_str_index = ['select endDate,closePrice from yuqerdata.yq_MktIdxwGet ',...
        'where ticker = ''%s'' and endDate>=''%s'' order by endDate'];
    x_index = fetchmysql(sprintf(sql_str_index,index_code,back_date),2);
    % 月度收益
    sql_str_month_return = ['select endDate,closeprice/precloseprice-1 from yuqerdata.yq_MktEquwAdjAfGet ',...
        'where ticker = "%s" and weekBeginDate>="%s" and tradeDays>0 order by endDate '];

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
    F = cell(T_symbols);
    parfor i = 1:T_symbols
        %读取股票数据
        x = fetchmysql(sprintf(sql_str_month_return,hs_300{i},begin_t),2);
        if size(x,1)<p_window*2
            F{i} = nan(T_tref,3);
            continue
        end
        sub_tref = x(:,1);
        x = cell2mat(x(:,2));
        x(isnan(x)) = 0;

        %[~,ia,ib] = intersect(tref,sub_tref);
        %temp_x = zeros(size(tref));
        %temp_x(ia) = x(ib);
        %temp_x = fillmissing(temp_x,'previous');
        %x = temp_x;
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
    %月度收益及两个指标
    return_df = F(:,1:3:end);
    speed_df = F(:,2:3:end);
    curve_df = F(:,3:3:end);

    %动量上涨、下跌分组
    ind_b_t = S39_methods.get_top_bottom(speed_df,30);
    r = zeros(T_tref,4);
    symbol_pool = cell(T_tref,4);
    for i = 1:T_tref
        %加速上涨
        ind1 = ind_b_t{i}{1};
        sub_x = curve_df(i,ind1);
        symbol_pool(i,1) = {hs_300(ind1(sub_x<0))};
        symbol_pool(i,2) = {hs_300(ind1(sub_x>0))};    
        if i <T_tref
            sub_r = return_df(i+1,ind1);
            sub_r1 = sub_r(sub_x<0);
            sub_r2 = sub_r(sub_x>0);
            if ~isempty(sub_r1)
                r(i+1,1) = mean(sub_r1);    
            end
            if ~isempty(sub_r2)
                r(i+1,2) = mean(sub_r2);    
            end
        end
        %加速下跌
        ind2 = ind_b_t{i}{2};
        sub_x2 = curve_df(i,ind2);
        symbol_pool(i,3) = {hs_300(ind2(sub_x2<0))};
        symbol_pool(i,4) = {hs_300(ind2(sub_x2>0))};  
        if i <T_tref
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
    end
    r(1,:) = 0;
    
    %r_index = cell2mat(x_index(:,2));
    %r_index(2:end) = r_index(2:end)./r_index(1:end-1)-1;
    %r_index(1) = 0;
    
    leg_str = {'top-down','top-up','bot-down','bot-up'};
    %多空  做多方法，做空指数
    %S39_methods.draw_figure(tref,r-r_index,'对冲指数',leg_str)
    S39_methods.draw_figure(tref,r,sprintf('周-对冲指数-%s',index_name),leg_str)
    leg_str2 = {'动量上涨加减速','动量下跌加减速'};
    S39_methods.draw_figure(tref,[r(:,1)-r(:,2),r(:,3)-r(:,4)],sprintf('周-加减速上涨下跌对冲-%s',index_name),leg_str2)
end
%{
%结果写入数据库,须保留
%var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};
sub_f1 = cell(size(tref));
sub_f2 = sub_f1;
for i = 1:T_tref
    sub_code_pool = symbol_pool(i,:);
    %more-less more-less
    sub_more1 = strjoin(sub_code_pool{1},',');
    sub_less1 = strjoin(sub_code_pool{2},',');
    sub_more2 = strjoin(sub_code_pool{3},',');
    sub_less2 = strjoin(sub_code_pool{4},',');
    sub_f1{i} = {tref{i},2,index_code,sub_more1,sub_less1}';
    sub_f2{i} = {tref{i},3,index_code,sub_more2,sub_less2}';
    
end
sub_f1 = [sub_f1{:}]';
sub_f2 = [sub_f2{:}]';

if ~isempty(sub_f1)
    datainsert_adair(tn,var_info,sub_f1);
end
if ~isempty(sub_f2)
    datainsert_adair(tn,var_info,sub_f2);
end
%}