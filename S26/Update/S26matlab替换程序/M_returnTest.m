clear

load re re

sql_str = 'SELECT tradedate,closeindex FROM yuqerdata.yq_index where symbol = ''399102'' order by tradedate';
x_1 = fetchmysql(sql_str,2);
tref_num = datenum(x_1(:,1));
ind = tref_num>=datenum(2017,1,1);
tref_num = tref_num(ind);
tref = x_1(ind,1);
index1 = cell2mat(x_1(ind,2));

T = length(tref);
y1 = zeros(T,1);
y2 = y1;
sql_str = 'select symbol,chgPct from yuqerdata.yq_dayprice where tradedate=''%s'' and chgPct is not null';

key_symbol = [];
for i = 1:13
    key_symbol = cat(1,key_symbol,re{i}(:,2));
end
key_symbol = unique(key_symbol);

key_symbol_all = cell(14,1);
for i = 1:13
    key_symbol_all{i} = unique(re{i}(:,2));
end

key_symbol_all{end} = key_symbol;

Y_re = cell(T,1);

parfor i = 1:T
    sub_x = fetchmysql(sprintf(sql_str,tref{i}),2);
    sub_y = cell2mat(sub_x(:,2));
    sub_re = zeros(1,15);
    for j = 1:14
        [~,ia] = intersect(sub_x(:,1),key_symbol_all{j});
        sub_re(j) = mean(sub_y(ia));
    end
    sub_re(end) = mean(sub_y);
    Y_re{i} = sub_re;
    sprintf('%d-%d',i,T)
    
end
Y_re = cellfun(@(x) x',Y_re,'UniformOutput',false);
Y_re = [Y_re{:}]';

title_str = cell(14,1);
for i = 1:13
    title_str{i} = sprintf('Rule-%d',i);
end
title_str{14} = '组合';
for i = 1:14
    figure;
    plot(tref_num,[index1/index1(1),cumprod(1+Y_re(:,[i,end]))],'LineWidth',2)
    datetick('x','yymm')
    legend({'yqer全A','财务风险组合','创业板综'},'NumColumns',3)
    title(title_str{i});
    setpixelposition(gcf,[430,368,1008,420]);

end
