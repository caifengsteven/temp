%{
交易行为因子的合成
上述交易行为因子在中信一级行业内进行因子去极值与因子标准化。
因子权重方面，我们滚动选取过去 12 期因子的 ICIR 值作为权重，加权形成交易行
为合成因子。

去极值 使用5median方法
因子标准化使用 zscore方法
%}

clear
%para
key_str = 'S32综合因子';
start_time = datetime;
print_sel = true;
tN_pool = {'S32.factor_apm','S32.factor_q','S32.s32_factor_inverse'};
tN_key_pool = {'apm','q','inver'};
tN_dir = [1,-1,-1];
tN_dir_str = {' ','-','-'};
window_ic = 12;

tN = 'S32.com_factor';
var_info = {'symbol','tradingdate','f_val'};

%ICIR
sql_str = 'select symbol,tradingdate,f_val from S32.rankIC_data order by tradingdate';
ic = fetchmysql(sql_str,2);
ic_t = unique(ic(:,2));
ic_value = zeros(length(ic_t),3);
icir = nan(size(ic_value));
for i = 1:3
    sub_ic = ic(strcmp(ic(:,1),tN_key_pool(i)),2:3);
    [~,ia,ib] = intersect(ic_t,sub_ic(:,1));
    ic_value(ia,i) = cell2mat(sub_ic(ib,2))*tN_dir(i);
    icir(:,i) = movmean(ic_value(:,i),[window_ic-1,0])./movstd(ic_value(:,i),[window_ic-1,0]);
end
icir(1:window_ic-1,:) = 1;
icir(icir<0) = 0;
%合成因子
%tref
for i = 1:length(tN_pool)
    sub_t = fetchmysql(sprintf('select distinct(tradingdate) from %s',tN_pool{i}),2);
    if eq(i,1)
        tref = intersect(sub_t,ic_t);
    else
        tref = intersect(sub_t,tref);
    end
end

[~,~,ia] = intersect(tref,ic_t,'stable');
icir = icir(ia,:);
icir = bsxfun(@rdivide,icir,sum(icir,2));
%截面处理
%update
t0 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
num0 = find(datenum(tref)>datenum(t0),1);
if isempty(num0)
    num0 = 1e9;
end
%icir = icir(ind);
%tref = tref(ind);
T = length(tref);

sql_str1 = 'select symbol,%sf_val from %s where tradingdate =''%s''';
%行业内去极值 使用5median方法
%行业内因子标准化使用 zscore方法
for i = num0:T
    x_indus = yq_methods.get_industry_class_2(tref{i});
    for j = 1:3
        sub_f = fetchmysql(sprintf(sql_str1,tN_dir_str{j},tN_pool{j},tref{i}),2);
        %industry
        [~,ia,ib] = intersect(sub_f(:,1),x_indus(:,1));
        sub_f = sub_f(ia,:);
        sub_f_value = cell2mat(sub_f(:,2));
        sub_indus = cell2mat(x_indus(ib,2));
        %industry del
        u_sub_indus = unique(sub_indus);
        for k = 1:length(u_sub_indus)
            sub_ind = eq(sub_indus,u_sub_indus(k));
            sub_sub_f_value = sub_f_value(sub_ind);
            %move outlier
            sub_sub_f_value1= factor_preprocessing.median_outlier_remove(sub_sub_f_value);      
            %zscore
            sub_sub_f_value1 = zscore(sub_sub_f_value1);
            sub_f_value(sub_ind) = sub_sub_f_value1;
            if print_sel
            	sprintf('%s：%d-%d-%d-%d',key_str,k,j,i,T)
            end
        end
        sub_f(:,2) = num2cell(sub_f_value);
        if eq(j,1)
            F = sub_f;
        else
            [~,ia,ib] = intersect(F(:,1),sub_f(:,1));
            F = [F(ia,:),sub_f(ib,2)];
        end
        
    end
    F_val = cell2mat(F(:,2:end));
    %加权
    F_f = bsxfun(@times,F_val,icir(i,:));
    %最终因子
    F_f = sum(F_f,2);
    %保存
    F_f = [F(:,1),F(:,1),num2cell(F_f)];
    F_f(:,2) = tref(i);
    %write to mysql
    if ~isempty(F_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,F_f);
        close(conna);            
    end  
end

end_time = datetime;
sprintf('Time used: %s',start_time-end_time)