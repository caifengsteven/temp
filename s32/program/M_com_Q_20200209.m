%{
我们取所有股票最近 10 个交易日的分钟行情数据，计算每只股票的情绪因子 Q。
%}
clear

print_sel = true;
tN = 'S32.factor_q';
var_info = {'symbol','tradingdate','f_val'};

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
%修正
tref_complete = fetchmysql('select distinct(tradingdate) from S32.factor_q',2);
[~,ia,ib] = intersect(month_cut_date2,tref_complete);

month_cut(ia,:)=[];
month_cut_date1(ia,:)=[];
month_cut_date2(ia,:)=[];

%获取最近10个交易日
T = size(month_cut_date1,1);
sql_str = 'select symbol,close,volume from ycz_min_history.`%s` where volume>0 order by tradingdate';
for i = 1:T
    sub_tref = tref(month_cut(i,2)-10+1:month_cut(i,2));
    x = cell(5000*60*4*10,3);
    x_id = 0;
    %按照顺序读入10个交易日数据
    for j = 1:length(sub_tref)
        temp_tref = sub_tref{j};
        temp_tref = temp_tref([1:4,6:7,9:10]);
        sub_x = fetchmysql(sprintf(sql_str,temp_tref),2);
        temp_l = size(sub_x,1);
        x(x_id+1:x_id+temp_l,:) = sub_x;
        x_id = x_id + temp_l;
    end
    x = x(1:x_id,:);
    symbol = unique(x(:,1));
    symbol2 = cellfun(@(x) x(3:end),symbol,'UniformOutput',false);
    sub_T = length(symbol);
    q = nan(sub_T,1);
    parfor j = 1:sub_T
        sub_x = cell2mat(x(strcmp(x(:,1),symbol(j)),2:3));
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
            sprintf('合成情绪因子：Complete: %d %d-%d',j,i,T)
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