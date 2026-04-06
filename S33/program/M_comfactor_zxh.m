clear

print_sel = true;
tN= 'S33.factor_zxh';
var_info = {'symbol','tradingdate','f_mv','f_reverse','f_std','f_change'};

%如果存在，将表格删除重新计算
%这里使用了一个简单方法，如果写过数据，就不再计算

sql_str = 'select * from %s limit 1';
temp = fetchmysql(sprintf(sql_str,tN),2);
if isempty(temp)
    window = 22;

    tref = yq_methods.get_tradingdate('2009-12-01','2020-01-13');
    tref_num = datenum(tref);
    %获取月底日期
    %last day for the month
    month_index = month(tref_num);
    month_cut = [0;find(diff(month_index))];
    month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
    month_cut_date1 = tref(month_cut(:,1));
    month_cut_date2 = tref(month_cut(:,2));
    month_trefnum = datenum(month_cut_date2);
    T_month = length(month_cut_date2);


    symbol = fetchmysql('select distinct(symbol) from yuqerdata.yq_dayprice',2);
    T = length(symbol);
    sql_str =[ 'select tradeDate,negMarketValue,chgPct,turnoverRate from ',...
        'yuqerdata.yq_dayprice where symbol = ''%s'' order by tradeDate'];

    parfor i = 1:T
        x = fetchmysql(sprintf(sql_str,symbol{i}),2);
        if size(x,1)<window*2
            continue
        end
        %逐个合成因子，最后合并
        %mv
        sub_x1 = x(:,[1,2]);
        sub_x1_v = cell2mat(sub_x1(:,2));
        del_ind = isnan(sub_x1_v);
        sub_x1(del_ind,:) = [];
        sub_x1_v(del_ind,:) = [];
        sub_x1_t = datenum(sub_x1(:,1));
        sub_f1 = nan(T_month,1);
        for j = 1:T_month
            ind = find(sub_x1_t<=month_trefnum(j),1,'last');
            if ~isempty(ind)
                sub_f1(j) = sub_x1_v(ind);
            end
        end

        %2 reverse 3 std
        sub_x2 = x(:,[1,3]);
        sub_x2_v = cell2mat(sub_x2(:,2));
        del_ind = isnan(sub_x2_v);
        sub_x2(del_ind,:) = [];
        sub_x2_v(del_ind,:) = [];
        sub_x2_t = datenum(sub_x2(:,1));
        sub_x2_f = nan(size(sub_x2_v));
        sub_x3_f = sub_x2_f;
        for j = window:length(sub_x2_t)
            temp1 = sub_x2_v(j-window+1:j);
            temp2 = cumprod(1+temp1)-1;
            sub_x2_f(j) = temp2(end);
            sub_x3_f(j) = std(temp1);
        end

        sub_f2 = nan(T_month,1);
        sub_f3 = sub_f2;
        for j = 1:T_month
            ind = find(sub_x2_t<=month_trefnum(j),1,'last');
            if ~isempty(ind)
                sub_f2(j) = sub_x2_f(ind);
                sub_f3(j) = sub_x3_f(ind);
            end
        end

        %4
        sub_x4 = x(:,[1,4]);
        sub_x4_v = cell2mat(sub_x4(:,2));
        del_ind = isnan(sub_x4_v);
        sub_x4(del_ind,:) = [];
        sub_x4_v(del_ind,:) = [];
        sub_x4_t = datenum(sub_x4(:,1));
        sub_x4_f = nan(size(sub_x4_v));
        for j = window:length(sub_x4_t)
            temp1 = sub_x4_v(j-window+1:j);
            sub_x4_f(j) = mean(temp1);
        end

        sub_f4 = nan(T_month,1);
        for j = 1:T_month
            ind = find(sub_x4_t<=month_trefnum(j),1,'last');
            if ~isempty(ind)
                sub_f4(j) = sub_x4_f(ind);
            end
        end

        sub_f = [month_cut_date2,month_cut_date2,num2cell([sub_f1,sub_f2,sub_f3,sub_f4])];
        sub_f(:,1) = symbol(i);
        del_ind = isnan(sum([sub_f1,sub_f2,sub_f3],2));
        sub_f = sub_f(~del_ind,:);    
        %write to mysql
        if ~isempty(sub_f)
            conna = mysql_conn();
            datainsert(conna,tN,var_info,sub_f);
            close(conna);            
        end

        if print_sel
            sprintf('合成中性化因子步骤： %d - %d',i,T)
        end
    end

end