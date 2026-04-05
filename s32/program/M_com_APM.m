%{
APM指标
%}
clear

print_sel = true;
tN = 'S32.factor_apm';
var_info = {'symbol','tradingdate','f_val'};

%读取时间
tref = yq_methods.get_tradingdate('2013-04-01','2020-01-13');
tref_num = datenum(tref);
T = size(tref,1);
sql_str1 = 'select hour(tradingdate)*100+minute(tradingdate),symbol,close from ycz_min_history.`%s` where close is not null order by tradingdate';
for i = 1:T
    sub_t = tref{i};
    sub_t = sub_t([1:4,6:7,9:10]);
    x = fetchmysql(sprintf(sql_str1,sub_t),2);
    %构建市场指数
    
    symbol = unique(x(:,1));
    symbol2 = cellfun(@(x) x(3:end),symbol,'UniformOutput',false);
    sub_T = length(symbol);
    apm_0 = nan(sub_T,1);
    for j = 1:sub_T
        sub_x = cell2mat(x(strcmp(x(:,1),symbol(j)),[1,3]));
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