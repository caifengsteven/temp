%流通股总市值 流通股数*月末收盘价
clear

tN = 'S29.factor_yuqer';
var_info = {'factor_name','pub_date','symbol','f_val'};
 
t0 = 'select pub_date from %s where factor_name = ''f16'' order by pub_date desc limit 1';
t0 = fetchmysql(sprintf(t0,tN),2);

if isempty(t0)
    t0 = '2003-03-01';
else
    t0 = t0{1};
end

m_d = {'03-31','06-30','09-30','12-31'};
year_num = 2003:year(now);

t_cut = zeros(length(year_num)*length(m_d),1);
tref_cut = cell(size(t_cut));
k = 1;
for i = 1:length(year_num)
    for j = 1:length(m_d)
        tref_cut{k} = sprintf('%d-%s',year_num(i),m_d{j});
        sub_t = datenum(sprintf('%d-%s',year_num(i),m_d{j}));
        t_cut(k) = sub_t;
        k = k + 1;
    end
end

ind = t_cut<=now & t_cut>datenum(t0);
t_cut = t_cut(ind);
tref_cut = tref_cut(ind);

if ~isempty(tref_cut)
    symbols = fetchmysql('select distinct(ticker) from yuqerdata.equget',2);
    T = length(symbols);
    sql_str = ['select tradedate,negMarketValue from yuqerdata.yq_dayprice ',...
        'where symbol = ''%s'' and tradedate>=''2003-01-01'' and negMarketValue is not null order by tradedate'];

    F = cell(T,1);
    parfor i = 1:T
        x = fetchmysql(sprintf(sql_str,symbols{i}),2);
        if isempty(x)
            continue
        end

        t = datenum(x(:,1));
        y = cell2mat(x(:,2));

        y_month = nan(size(t_cut));
        for j = 1:length(t_cut)
            ia = t<t_cut(j);
            if any(ia)
                temp = y(ia);
                y_month(j) = temp(end);            
            end
        end
        nan_ind = isnan(y_month);
        sub_x = [tref_cut,tref_cut,num2cell(y_month)];
        sub_x(:,1) = symbols(i);
        sub_x(nan_ind,:) = [];    
        F(i) = {sub_x'};
        sprintf('Complete: %d-%d',i,T)

    end
    F = [F{:}]';
    F  = F(:,[1,2,1,3]);
    F(:,1) = {'f16'};
    %datainset to mysql
    if ~isempty(F)
        conna = mysql_conn();
        datainsert(conna,tN,var_info,F);
        close(conna);            
    end
end
