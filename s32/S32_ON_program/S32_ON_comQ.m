%{
我们取所有股票最近 10 个交易日的分钟行情数据，计算每只股票的情绪因子 Q。
%}
clear
key_str = 'S32合成情绪因子';
start_time = datetime;
print_sel = true;
tN = 'S32.factor_q';
var_info = {'symbol','tradingdate','f_val'};

%读取时间
tref1 = yq_methods.get_tradingdate('2013-04-01',datestr(now,'yyyy-mm-dd'));
tref2 = yq_methods.get_tradingdate_future(tref1{end});
tref = [tref1;tref2(2)];
tref_num = datenum(tref);
%获取月底日期
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

%chek 计算完毕的
temp_sql = 'select tradingdate from %s order by tradingdate desc limit 1';
tref_complete = fetchmysql(sprintf(temp_sql,tN),2);
num0 = find(strcmp(month_cut_date2,tref_complete))+1;

%去掉计算完毕的
%获取最近10个交易日
T = size(month_cut_date1,1);
sql_str = 'select symbol,close,volume from ycz_min_history.`%s` where volume>0 order by tradingdate';
for i = num0:T
    sub_tref = tref(month_cut(i,2)-10+1:month_cut(i,2));
    x = cell(size(sub_tref));
    %按照顺序读入10个交易日数据
    parfor j = 1:length(sub_tref)
        temp_tref = sub_tref{j};
        temp_tref = temp_tref([1:4,6:7,9:10]);
        sub_x = fetchmysql(sprintf(sql_str,temp_tref),2);
        temp_l = size(sub_x,1);
        x(j) = {sub_x'};
    end
    x = [x{:}]';
    symbol = unique(x(:,1));
    symbol2 = cellfun(@(x) x(3:end),symbol,'UniformOutput',false);
    sub_T = length(symbol);
    q = nan(sub_T,1);
    x_symbol = x(:,1);
    x_v = cell2mat(x(:,2:3));
    parfor j = 1:sub_T
        %sub_ind = strcmp(x(:,1),symbol(j));
        %sub_x = x(sub_ind,2:3);
        %sub_x = cell2mat(sub_x);
        sub_ind = strcmp(x_symbol,symbol(j));
        sub_x = x_v(sub_ind,:);
        %少于1天数据，不予计算
        if size(sub_x,1)<= 60*4
            continue
        end
        sub_x_r = zeros(size(sub_x(:,1)));
        sub_x_r(2:end) = sub_x(2:end,1)./sub_x(1:end-1,1)-1;
        sub_s = abs(sub_x_r)./sqrt(sub_x(:,2));
        [sub_s,ia] = sort(sub_s,'descend');
        sub_x = sub_x(ia,:);
        sub_x_v = cumsum(sub_x(:,2));
        sub_x_v = sub_x_v./sub_x_v(end);
        
        sub_id_smart = sub_x_v<=0.2;
        
        if not(any(sub_id_smart))
            q(j) = 0;
            continue
        end
        vwap_smart = sum(sub_x(sub_id_smart,1).*sub_x(sub_id_smart,2))/sum(sub_x(sub_id_smart,2));
        vwap_all = sum(sub_x(:,1).*sub_x(:,2))/sum(sub_x(:,2));
        %计算Q
        
        q(j) = vwap_smart/vwap_all;
        if print_sel
            sprintf('%s: %d %d-%d',key_str,j,i,T)
        end
    end
    
    sub_f = [symbol2,symbol2,num2cell(q)];
    sub_f(:,2) = month_cut_date2(i);
    temp = ~isnan(q);
    sub_f = sub_f(temp,:);
    %保存
     %write to mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);            
    end   
    
end
end_time = datetime;
sprintf('Time used: %s',start_time-end_time)