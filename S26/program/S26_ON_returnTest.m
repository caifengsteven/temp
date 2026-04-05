clear
tn = 'S37.S26_bac';
var_info = {'tradingdate','y1','y2','y3'};

sql_str = 'SELECT tradedate,closeindex FROM yuqerdata.yq_index where symbol = ''399102'' and tradedate>=''2010-01-01'' order by tradedate';
x_1 = fetchmysql(sql_str,2);
tref_num = datenum(x_1(:,1));
ind = tref_num>=datenum(2011,1,1);
tref_num = tref_num(ind);
tref = x_1(ind,1);
index1 = cell2mat(x_1(ind,2));

T = length(tref);
y1 = zeros(T,1);
y2 = y1;
sql_str = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradedate=''%s'' and chgPct is not null';

key_symbol = [];
ind = [1,2,4,7,8,10];
sql_str1 = 'select symbol from S26.S26_result where rule_name = %d order by tradingdate';
for i = 1:length(ind)    
    key_symbol = cat(1,key_symbol,fetchmysql(sprintf(sql_str1,ind(i)),2));
end
key_symbol = unique(key_symbol);

x0 = fetchmysql(sprintf('select * from %s order by tradingdate',tn),2);
if isempty(x0)
    t0 = '2001-01-01';
else
    t0 = x0{end,1};
end
num0 = find(strcmp(tref,t0));
if isempty(num0)
    num0 = 0;
end
parfor i = num0+1:T
    sub_x = fetchmysql(sprintf(sql_str,tref{i}),2);
    sub_y = cell2mat(sub_x(:,2));
    y1(i) = mean(sub_y);
    [~,ia] = intersect(sub_x(:,1),key_symbol);
    y2(i) = mean(sub_y(ia));
    sprintf('%d-%d',i,T)
end

re = [tref,num2cell([index1,y1,y2])];
re = re(num0+1:end,:);
if ~isempty(re)
    datainsert_adair(tn,var_info,re)
end

re = [x0;re];

index1 = cell2mat(re(:,2));
y1 =cell2mat(re(:,3));
y2 = cell2mat(re(:,4));

figure;
plot(tref_num,[index1/index1(1),cumprod(1+[y2,y1])],'LineWidth',2)
datetick('x','yymm')
legend({'yqer全A','财务风险组合','创业板综'},'NumColumns',3)
title('组合—update');
setpixelposition(gcf,[430,368,1008,420]);


