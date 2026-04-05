%合成CVaR因子
%左侧，右侧，成交量加权CvaR的左侧、右侧，共计4个因子的单日值
%预测者日内分钟数据计算日内收益，优矿昨日收盘数据用于构建第一个时间点的收益

%程序运行时间较长，8核心并行用了12+个小时，中间958、959中间matlab平台崩溃，从断点处重新
%运行后一切正常

print_sel = true;

alpha = 0.05;
%tN = 'S32.factor_index_min';
%var_info = {'tradingdate','f_val1','f_val2'};
tN = 'S33.factor_cvar';
var_info = {'symbol','tradingdate','f_val1','f_val2','f_val3','f_val4'};

tref_complete = fetchmysql(sprintf('select distinct(tradingdate) from %s',tN),2);
%读取时间
tref = yq_methods.get_tradingdate('2015-12-01','2020-01-13');
tref = setdiff(tref,tref_complete);
tref_num = datenum(tref);

T = size(tref,1);
sql_str1 = ['select symbol,close,volume from ycz_min_history.`%s` where ',...
    'close is not null and volume is not null order by tradingdate']; %预测者
sql_str2 = ['select symbol,actPreClosePrice from yuqerdata.yq_dayprice ',...
    'where tradeDate = ''%s'' and actPreClosePrice is not null']; %优矿

for i = 1:T
    sub_t = tref{i};
    sub_t = sub_t([1:4,6:7,9:10]);
    x = fetchmysql(sprintf(sql_str1,sub_t),2);
    %构建市场指数
    x_pre = fetchmysql(sprintf(sql_str2,sub_t),2);
    
    symbol = unique(x(:,1));
    symbol2 = cellfun(@(x) x(3:end),symbol,'UniformOutput',false);
    temp = unique(x_pre(:,1));
    [symbol2,ia] = intersect(symbol2,temp);
    symbol = symbol(ia);
    
    sub_T = length(symbol);
    sub_f = nan(sub_T,4);
    parfor j = 1:sub_T
        ind = strcmp(x(:,1),symbol(j));
        sub_x_pre = x_pre(strcmp(x_pre(:,1),symbol2(j)),2);
        sub_x_symbol = cell2mat(x(ind,2:3));
        if size(sub_x_symbol,1)<120
            continue
        end
        sub_x_t = cell2mat(x(ind,1));
        if isempty(sub_x_pre)
            sub_x_pre = sub_x_symbol(1,1);
        else
            sub_x_pre = cell2mat(sub_x_pre);
        end
        sub_x = [sub_x_pre;sub_x_symbol(:,1)];
        sub_r = zeros(size(sub_x));
        sub_r(2:end) = sub_x(2:end)./sub_x(1:end-1)-1;
        sub_r = sub_r(2:end);
        sub_r(isnan(sub_r)) = 0;
        sub_r(isinf(sub_r)) = 0;
        
        sub_x_volume = sub_x_symbol(:,2);
        [~,CVaR1,~,CVaR2] = var_cvar(sub_r,alpha);
        [~,CVaR3,~,CVaR4] = var_cvar(sub_r.*sub_x_volume,alpha);
        sub_f(j,:) = [CVaR1,CVaR2,CVaR3,CVaR4];
        if print_sel
            sprintf('CVAR and VCVAR indicator cal: %d-%d %d-%d',j,sub_T,i,T)
        end
    end
    sub_f1_f = [symbol2,symbol2,num2cell(sub_f)];
    sub_f1_f(:,2) = tref(i);
    del_ind = isnan(sum(sub_f,2));
    sub_f1_f(del_ind,:) = [];
    
     %write to mysql
    if ~isempty(sub_f1_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f1_f);
        close(conna);            
    end

    
end