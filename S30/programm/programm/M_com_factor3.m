%month_data
%营业收入_TTM / 总市值

clear
tN = 'S30.F3_month';
var_info = {'symbol','tradingdate','f_val'};
symbol = fetchmysql('select distinct(symbol) from yuqerdata.yq_dayprice',2);

%tref = fetchmysql('select distinct(tradeDate) from yuqerdata.yq_dayprice order by tradeDate',2);
load tref
tref_num = datenum(tref);

sel_ind = tref_num>=datenum(2003,1,1);
tref = tref(sel_ind);
tref_num = tref_num(sel_ind);
%last day for the month
month_index = month(tref_num);
month_cut = [0;find(diff(month_index))];
month_cut = [month_cut(1:end-1)+1,month_cut(2:end)];
month_cut_date1 = tref(month_cut(:,1));
month_cut_date2 = tref(month_cut(:,2));

T_month = length(month_cut_date2);
T = length(symbol);
%股东权益合计(不含少数股东权益)_最新财报
F = fetchmysql('select * from S30.F3_season order by pub_date',2);
sql_str = 'select tradeDate,marketvalue from yuqerdata.yq_dayprice where symbol = ''%s'' order by tradeDate';
for i = 1:T
    sub_F = F(strcmp(F(:,1),symbol(i)),:);
    sub_mv = fetchmysql(sprintf(sql_str,symbol{i}),2);
    
    if isempty(sub_F) || isempty(sub_mv)
        continue
    end
    
    [sub_F_filling,sub_F_tref] = yq_methods.filling_data(month_cut_date2,sub_F(:,2),cell2mat(sub_F(:,end)));
    
    [sub_mv_sel,sub_mv_tref] = yq_methods.find_near_data(month_cut_date2,sub_mv(:,1),cell2mat(sub_mv(:,2)));
    
    [sub_tref,ia,ib] = intersect(sub_F_tref,sub_mv_tref);
    if isempty(sub_tref)
        continue
    end
    sub_f = sub_F_filling(ia,:)./sub_mv_sel(ib);
    nan_ind = isnan(sub_f);
    sub_f = [cellstr(datestr(sub_tref(~nan_ind),'yyyy-mm-dd')),num2cell(sub_f(~nan_ind))];
    sub_f = sub_f(:,[1,1:end]);
    sub_f(:,1) = symbol(i);
    
    %to mysql
    if ~isempty(sub_f)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,sub_f);
        close(conna);            
    end
    sprintf('合成细分因子3 Complete: %d-%d',i,T)
end





