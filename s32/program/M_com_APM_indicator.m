%{
APM指标
构建市场指数
预测者的第一分钟的数据，需要每日数据的收盘数据来计算收益率
只计算月度频率的数据

可以升级为计算日度数据
%}
clear

print_sel = true;
tN = 'S32.factor_index_min';
var_info = {'tradingdate','f_val1','f_val2'};

window = 60;
%读取时间
tref = yq_methods.get_tradingdate('2013-04-01','2020-01-13');
tref_num = datenum(tref);
%获取月底日期
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

tref = month_cut_date2;
T = size(tref,1);
sql_str1 = 'select tradingdate,symbol,close from ycz_min_history.`%s` where close is not null order by tradingdate';
sql_str2 = 'select symbol,actPreClosePrice from yuqerdata.yq_dayprice where tradeDate = ''%s'' ';
sql_str3 = 'select ticker from   yuqerdata.st_info where tradedate =''%s''';
sql_str4 = ['select ticker,listDate from yuqerdata.equget where listStatusCd !=''UN''',...
    'and listDate is not null']; 
symbol_info = fetchmysql(sql_str4,2);
symbol_listdate = datenum(symbol_info(:,2));
parfor i = 1:T
    sub_t = tref{i};
    sub_t = sub_t([1:4,6:7,9:10]);
    x = fetchmysql(sprintf(sql_str1,sub_t),2);
    %构建市场指数
    x_pre = fetchmysql(sprintf(sql_str2,sub_t),2);
    
    symbol = unique(x(:,2));
    symbol2 = cellfun(@(x) x(3:end),symbol,'UniformOutput',false);
    sub_T = length(symbol);
    apm_0 = nan(sub_T,1);
    x2 = x;
    for j = 1:sub_T
        ind = strcmp(x(:,2),symbol(j));
        sub_x_pre = x_pre(strcmp(x_pre(:,1),symbol2(j)),2);
        temp = cell2mat(x(ind,3));
        if isempty(sub_x_pre)
            sub_x_pre = temp(1);
        else
            sub_x_pre = cell2mat(sub_x_pre);
        end
        sub_x = [sub_x_pre;temp];
        sub_r = zeros(size(sub_x));
        sub_r(2:end) = sub_x(2:end)./sub_x(1:end-1)-1;
        x2(ind,end) = num2cell(sub_r(2:end));
        if print_sel
            sprintf('APM indicator:cal return step %d-%d %d-%d',j,sub_T,i,T)
        end
    end
    
    %st
    st = fetchmysql(sprintf(sql_str3,tref{i}),2);
    st = cellfun(@str2double,st,'UniformOutput',false);
    st = cellfun(@(x) sprintf('%0.6d',x),st,'UniformOutput',false);
    %上市未满 60 日的新股
    symbol_remain = symbol_info(datenum(tref{i})-symbol_listdate>window,1);
    sub_t_u = unique(x2(:,1));
    T_sub_t_u = length(sub_t_u);
    x3 = zeros(T_sub_t_u,2);
    for j = 1:T_sub_t_u
        sub_x = x2(strcmp(x2(:,1),sub_t_u(j)),2:3);
        sub_x(:,1) = cellfun(@(x) x(3:end),sub_x(:,1),'UniformOutput',false);
        sub_x_v = cell2mat(sub_x(:,2));
        temp1 = mean(sub_x_v(~isnan(sub_x_v)&~isinf(sub_x_v)));
        
        [~,ia]= intersect(sub_x(:,1),st);
        sub_x(ia,:) = [];
        sub_x_v(ia,:) = [];
        [~,ia] = intersect(sub_x(:,1),symbol_remain);
        sub_x = sub_x(ia,:);
        sub_x_v = sub_x_v(ia,:);
        temp2 = mean(sub_x_v(~isnan(sub_x_v)&~isinf(sub_x_v)));
        x3(j,:) = [temp1,temp2];
        if print_sel
            sprintf('APM indicator:Static step %d-%d %d-%d',j,T_sub_t_u,i,T)
        end
    end
    sub_f = [sub_t_u,num2cell(x3)];

    %保存
     %write to mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);            
    end   
    
end