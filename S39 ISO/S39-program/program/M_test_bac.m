clear
[~,~,s]= xlsread('m1_000300.csv');
tref = s(2:end,1);
tref = cellstr(datestr(datenum(tref),'yyyy-mm-dd'));
symbol_m = s(2:end,2);
symbol_l = s(2:end,3);

T = length(tref);
sql_month_r = ['select ticker,chgPct from yuqerdata.MktEqumAdjAfGet ',...
    'where endDate = ''%s'''];
%x = fetchmysql();
y = cell(T,1);
parfor i = 1:T-1
    sub_m = strsplit(symbol_m{i},',');
    sub_l = strsplit(symbol_l{i},',');
    
    x = fetchmysql(sprintf(sql_month_r,tref{i+1}),2);
    [~,ia,ib] = intersect(sub_m,x(:,1));
    r1 = zeros(size(sub_m(:,1)));
    r1(ia) = cell2mat(x(ib,2));
    
    [~,ia,ib] = intersect(sub_l,x(:,1));
    r2 = zeros(size(sub_l(:,1)));
    r2(ia) = cell2mat(x(ib,2));
    
    
    y{i} = [mean(r1),mean(r2)]';
end