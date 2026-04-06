%合成CVaR因子
%左侧，右侧，成交量加权CvaR的左侧、右侧，共计4个因子的单日值
%预测者日内分钟数据计算日内收益，优矿昨日收盘数据用于构建第一个时间点的收益
%month 每日计算左侧 5%CVaR 值，每月末对当月所有交易日的 CVaR 值
%求算术平均，得到月频的 CVaR 因子。
print_sel = true;
key_str = 'S33合成月度因子1';
tN = 'S33.factor_cvar';

tN_month = 'S33.factor_cvar_month';
var_info = {'symbol','tradingdate','f_val1','f_val2','f_val3','f_val4'};

%读取时间
t1 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN_month),2);
t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');
t2 = datestr(now,'yyyy-mm-dd');
tref1 = yq_methods.get_tradingdate(t1,t2);
tref2 = yq_methods.get_tradingdate_future(tref1{end});
tref = [tref1;tref2(2)];%顺延一天便于确认最后一天是否为月底
tref_num = datenum(tref);

%月度数据
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
if length(month_cut)<2
    T = 0;
    sprintf('%s:已经是最新日期，无需更新',key_str)
else
    month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
    month_cut_date1 = tref(month_cut(:,1));
    month_cut_date2 = tref(month_cut(:,2));
    %过滤数据
    T = size(month_cut_date2,1);
    sql_str1 = ['select symbol,f_val1,f_val2,f_val3,f_val4 from %s where ',...
        'tradingdate>=''%s'' and tradingdate <=''%s'' ']; %获取数据
end



for i = 1:T
    
    sub_t1 = month_cut_date1{i};
    sub_t2 = month_cut_date2{i};
    
    %获取月度数据
    x = fetchmysql(sprintf(sql_str1,tN,sub_t1,sub_t2),2);
    sub_symbol = unique(x(:,1));
    
    T_sub_symbol = length(sub_symbol); 
    sub_f = nan(T_sub_symbol,4);
    %构建月度因子
    parfor j = 1:T_sub_symbol
        sub_ind = strcmp(x(:,1),sub_symbol(j));
        sub_x = cell2mat(x(sub_ind,2:end));
        if size(sub_x,1) < 5
            continue
        end
        sub_f(j,:) = mean(sub_x,1);
        if print_sel
            sprintf('%s: %d-%d %d-%d',key_str,j,T_sub_symbol,i,T)
        end
    end
    
    sub_f1_f = [sub_symbol,sub_symbol,num2cell(sub_f)];
    sub_f1_f(:,2) = {sub_t2};
    del_ind = isnan(sum(sub_f,2));
    sub_f1_f(del_ind,:) = [];
    
     %write to mysql
    if ~isempty(sub_f1_f)
        conna = mysql_conn();
        datainsert(conna,tN_month,var_info,sub_f1_f);
        close(conna);            
    end

    
end