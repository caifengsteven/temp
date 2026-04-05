%{
目标将国泰安净利润数据转换为单季数据并保存

update 
加快计算速度
增加同比增速计算

%}

clear

%逐个转换并写入数据库
%股票代码，截止日期，报告发布日期，净利润（1个季度）,同比增速，净利润（1个季度）上期,同比增速上期，净利润（1个季度）去年同期
sql_str1 = ['select symbol,date2,date1,NI_val from gtadata.nincomedata_ycz where NI_val is not null ',...
    'and date1>''2002-01-01'' order by symbol,date1' ];
x = fetchmysql(sql_str1,2);
x(:,1) = cellfun(@(x) sprintf('%0.6d.SHSE',x),x(:,1),'UniformOutput',false);
symbol = unique(x(:,1));
T = length(symbol);

re = cell(size(x,1),8);
re_ind = 0;


for i = 1:T
    sub_x1=x(strcmp(x(:,1),symbol(i)),:);
    sub_x1_t = datenum(sub_x1(:,3));
    sub_x1_y = year(sub_x1_t);
    sub_x1_y_u = unique(sub_x1_y,'stable');
    temp = zeros(size(sub_x1(:,1)));
    for j = 1:length(sub_x1_y_u)
        ia = eq(sub_x1_y,sub_x1_y_u(j));
        sub_sub_x1 = cell2mat(sub_x1(ia,4));        
        temp(ia) = [sub_sub_x1(1);diff(sub_sub_x1)];        
    end
    sub_re = [sub_x1(:,[1,3,2]),num2cell(temp)];
    %计算同比增速
    sub_t2 = size(sub_re,1);
    if sub_t2>=5
        temp = cell2mat(sub_re(:,end));
        temp1 = nan(sub_t2,1);
        temp1(5:end) = (temp(5:end)-temp(1:end-4))./(temp(1:end-4));
        %上期为负数的必须去掉
        temp_ind = 5:length(temp1);
        temp1(temp_ind(temp(1:end-4)<0)) = 0;
        
        sub_re = [sub_re,num2cell(temp1)];
    else
        sub_re = [sub_re,num2cell(nan(sub_t2,1))];
    end
    %将上期的数据和增速加入数据结构，方便后期检索
    temp1 = cell(sub_t2,3);
    temp1(2:end,1:2) = sub_re(1:end-1,end-1:end);
    temp1(5:end,3) = sub_re(1:end-4,4);
    sub_re = [sub_re,temp1];
    
    sub_t = size(sub_re,1);
    re(re_ind+1:re_ind+sub_t,:) = sub_re;
    re_ind = re_ind+sub_t;
    sprintf('%d-%d',i,T)
end
re = re(1:re_ind,:);
re = [{'symbol','endDate','pubDate','nincome1','nin_rate1','nincome0','nin_rate0','nincome_5'};re];

xlswrite('nincome.xlsx',re);

