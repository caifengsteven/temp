clear
key_str = '写入历史收益数据';
tn = 'S37.S39_result';
var_info = {'tradingdate', 'method_ID', 'index_code', 'more_r', 'less_r'};
tn_symbol = 'S37.symbol_pool_S39';

index_pool = {'000300','000905','000001'};
index_info = {'沪深300','中证500','上证综指'};
t0 = yq_methods.get_table_end_date(tn);
if isempty(t0)
    t0 = '2008-06-06';
end
tref = yq_methods.get_tradingdate(t0);
tref = tref(2:end);
if isempty(tref)
    sprintf('无需更新，已经是最新 %s',key_str)
    return
end
T = length(tref);

sql_str_r = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradeDate = ''%s''';
sql_str_t = ['select tradingdate from %s where method_ID = %d and index_code=''%s'' ',...
    'and tradingdate<''%s'' order by tradingdate desc limit 1'];
sql_str_symbol = ['select more_r,less_r from %s where method_ID = %d and index_code=''%s'' ',...
    'and tradingdate=''%s'''];
r_re1 = cell(T,1);
parfor i = 1:T
    sub_r = fetchmysql(sprintf(sql_str_r,tref{i}),2);
    sub_re1 = cell(3,1);
    for j = 1:3%3个方法
        sub_re2 = cell(3,1);
        for k = 1:3%3个股票池
            t = fetchmysql(sprintf(sql_str_t,tn_symbol,j,index_pool{k},tref{i}),2);
            if isempty(t)
                continue
            else
                t = t{1};
            end
            sub_symbol = fetchmysql(sprintf(sql_str_symbol,tn_symbol,j,index_pool{k},t),2);
            sub_symbol_m = strsplit(sub_symbol{1},',');
            sub_symbol_l = strsplit(sub_symbol{2},',');
            [~,ia,ib] = intersect(sub_r(:,1),sub_symbol_m);
            sub_r_m = zeros(size(sub_symbol_m(:,1)));
            sub_r_m(ib) = cell2mat(sub_r(ia,2));
            [~,ia,ib] = intersect(sub_r(:,1),sub_symbol_l);
            sub_r_l = zeros(size(sub_symbol_l(:,1)));
            sub_r_l(ib) = cell2mat(sub_r(ia,2));
            
            sub_re2{k} = {tref{i},j,index_pool{k},mean(sub_r_m),mean(sub_r_l)}';
        end
        sub_re1{j} = [sub_re2{:}];
    end
    r_re1{i} = [sub_re1{:}];
    sprintf('%s %d-%d',key_str,i,T)
end
r_re1 = [r_re1{:}]';
if ~isempty(r_re1)
    datainsert_adair(tn,var_info,r_re1);
end
