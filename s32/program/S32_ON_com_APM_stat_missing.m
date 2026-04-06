%{
合成APM指标
%}
clear

key_str = 'S32合成STAT指标';
m_start_time = datetime;
print_sel = true;
tN= 'S32.factor_delta';
var_info = {'symbol','tradingdate','f_val'};

tN1 = 'S32.factor_symbolreturn_apm';
tN2 = 'S32.factor_indexreturn_apm';

window1 = 20;
%读取指数数据
sql_str0 = 'select tradingdate,f_am2,f_pm2 from %s where tradingdate>=''%s'' and tradingdate<=''%s'' order by tradingdate';

%x_index = 'select tradingdate,f_am2,f_pm2 from %s order by tradingdate';
%x_index = fetchmysql(sprintf(x_index,tN2),2);

%symbol = fetchmysql(sprintf('select distinct(symbol) from %s',tN1),2);
%sql_str1 = 'select tradingdate,f_am,f_pm from %s where symbol = ''%s'' order by tradingdate';
sql_str1 = 'select symbol,tradingdate,f_am,f_pm from %s where tradingdate>= ''%s'' and tradingdate<=''%s'' order by tradingdate';

%还没有计算的日期
%t1 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
%t1 = datestr(datenum(t1)+1,'yyyy-mm-dd');%从下一个日期开始
%t2 = datestr(now,'yyyy-mm-dd');%当前时间（截至时间）
%tref = yq_methods.get_tradingdate(t1,t2);%没有计算过的时间

tref = yq_methods.get_tradingdate('2020-08-01','2022-08-26');%没有计算过的时间
tref1 = fetchmysql(sprintf('select distinct(tradingdate) from %s where tradingdate>="2020-08-01"',tN),2);
tref = setdiff(tref,tref1);

T = length(tref);

for i = 1:T
    
    sub_tref = yq_methods.get_tradingdate('2000-01-01',tref{i});
    sub_tref = sub_tref(end-window1+1:end);
    %载入数据
    x_index = fetchmysql(sprintf(sql_str0,tN2,sub_tref{1},sub_tref{end}),2);
    sub_x = fetchmysql(sprintf(sql_str1,tN1,sub_tref{1},sub_tref{end}),2);
    
    symbol = sub_x(strcmp(sub_x(:,2),sub_tref(end)),1);
    T_symbol = length(symbol);
    stat_v = nan(T_symbol,1);
    parfor j = 1:T_symbol
        warning('off')
        ind2 = strcmp(sub_x(:,1),symbol(j));
        
        sub_x_index = x_index;
        sub_sub_x = sub_x(ind2,2:end);
        
        [~,ia,ib] = intersect(sub_x_index(:,1),sub_sub_x(:,1),'stable');
        sub_x_window = [sub_sub_x(ib,:),sub_x_index(ia,2:end)];
        
        sub_x_window = cell2mat(sub_x_window(:,2:end));
        %del_nan
        sub_x_window(isnan(sum(sub_x_window,2)),:) = [];
        if size(sub_x_window,1)<window1  %小于window1不计算
            continue
        end
        
        sub_x_test = [[sub_x_window(:,1);sub_x_window(:,2)],[sub_x_window(:,3);sub_x_window(:,4)]]; %单股y 指数x
        %cal r
        %sub_x_test 1
        [~,~,resi] = regress(sub_x_test(:,1),[ones(size(sub_x_test)),sub_x_test(:,2)]);
        delta = resi(1:window1)-resi(window1+1:end); %上午-下午
        stat_v(j) = mean(delta)/(std(delta)/sqrt(window1));
        if print_sel
            sprintf('%s：%d-%d %d-%d',key_str,j,T_symbol,i,T)
        end
    end
    
    sub_f = [symbol,symbol,num2cell(stat_v)];
    del_ind = isnan(stat_v);
    sub_f(del_ind,:) = [];
    sub_f(:,2) = tref(i);
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);
    end
    
    if print_sel
        sprintf('com delta step: %d-%d',i,T)
    end
end
warning('on')
m_end_time = datetime;
sprintf('Time used %s',m_end_time-m_start_time)