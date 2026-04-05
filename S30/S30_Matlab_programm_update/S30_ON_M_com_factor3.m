%month_data
%营业收入_TTM / 总市值

clear
key_str = '合成细分因子3';
tN = 'S30.F3_month';
var_info = {'symbol','tradingdate','f_val'};
symbol = fetchmysql('select distinct(symbol) from yuqerdata.yq_dayprice',2);
t0 = fetchmysql(sprintf('select tradingdate from %s order by tradingdate desc limit 1',tN),2);
%last day for the month
month_cut_date = yq_methods.get_month_data();
month_cut_date_num = datenum(month_cut_date);
ind = month_cut_date_num>=datenum(t0);
month_cut_date = month_cut_date(ind);
month_cut_date_num = month_cut_date_num(ind);
T_month = length(month_cut_date);
if eq(T_month,1)
    sprintf('%s:Complete!',key_str)
    return
end
symbol = yq_methods.get_symbol_A();
T = length(symbol);
%股东权益合计(不含少数股东权益)_最新财报
F = fetchmysql('select * from S30.F3_season order by pub_date',2);
sql_str = 'select tradeDate,marketvalue from yuqerdata.yq_dayprice where symbol = ''%s'' and tradeDate >=''%s'' order by tradeDate';
X = cell(T,1);
parfor i = 1:T
    sub_F = F(strcmp(F(:,1),symbol(i)),:);
    sub_mv = fetchmysql(sprintf(sql_str,symbol{i},t0{1}),2);
    
    if isempty(sub_F) || isempty(sub_mv)
        continue
    end
    
    [sub_F_filling,sub_F_tref] = yq_methods.filling_data(month_cut_date,sub_F(:,2),cell2mat(sub_F(:,end)));
    
    [sub_mv_sel,sub_mv_tref] = yq_methods.find_near_data(month_cut_date,sub_mv(:,1),cell2mat(sub_mv(:,2)));
    
    [sub_tref,ia,ib] = intersect(sub_F_tref,sub_mv_tref);
    if isempty(sub_tref)
        continue
    end
    sub_f = sub_F_filling(ia,:)./sub_mv_sel(ib);
    nan_ind = isnan(sub_f);
    sub_f = [cellstr(datestr(sub_tref(~nan_ind),'yyyy-mm-dd')),num2cell(sub_f(~nan_ind))];
    sub_f = sub_f(:,[1,1:end]);
    sub_f(:,1) = symbol(i);
    if isempty(sub_f)
        continue
    end
    sub_f = sub_f(datenum(sub_f(:,2))>datenum(t0),:);
    X{i} = sub_f';
    sprintf('%s:Complete: %d-%d',key_str,i,T)
end
X = [X{:}]';
%to mysql
if ~isempty(X)
    conna = mysql_conn();
    datainsert(conna,tN,var_info,X);
    close(conna);            
end




