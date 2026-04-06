%{
合成APM指标
%}
clear

clear

print_sel = true;
tN= 'S32.factor_delta';
var_info = {'symbol','tradingdate','f_val'};

tN1 = 'S32.factor_symbolreturn_apm';
tN2 = 'S32.factor_indexreturn_apm';

window = 20;
%读取指数数据
x_index = 'select tradingdate,f_am2,f_pm2 from %s order by tradingdate';
x_index = fetchmysql(sprintf(x_index,tN2),2);

symbol = fetchmysql(sprintf('select distinct(symbol) from %s',tN1),2);
sql_str1 = 'select tradingdate,f_am,f_pm from %s where symbol = ''%s'' order by tradingdate';
T = length(symbol);

parfor i = 1:T
    warning('off')
    sub_x = fetchmysql(sprintf(sql_str1,tN1,symbol{i}),2);
    [~,ia,ib] = intersect(x_index(:,1),sub_x(:,1),'stable');
    
    sub_x = [sub_x(ib,:),x_index(ia,2:end)];%单股，指数
    
    if size(sub_x,1)<=window
        continue
    end
    sub_x_T = size(sub_x,1);
    stat_v = nan(sub_x_T,1);
    for j = window:sub_x_T
        sub_window = j-window+1:j;
        sub_x_window = cell2mat(sub_x(sub_window,2:end));
        
        sub_x_test = [[sub_x_window(:,1);sub_x_window(:,2)],[sub_x_window(:,3);sub_x_window(:,4)]]; %单股y 指数x
        %cal r
        %sub_x_test 1
        [~,~,resi] = regress(sub_x_test(:,1),[ones(size(sub_x_test)),sub_x_test(:,2)]);
        delta = resi(1:window)-resi(window+1:end); %上午-下午
        stat_v(j) = mean(delta)/(std(delta)/sqrt(window));
    end
    sub_f = [sub_x(:,[1,1]),num2cell(stat_v)];
    del_ind = isnan(stat_v);
    sub_f(del_ind,:) = [];
    sub_f(:,1) = symbol(i);
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
