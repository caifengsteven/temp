%{
盘中异动现象的短期收益观察
不同阈值个股异动信号触发后dt时间段收益统计
%}
clear

tb_name = 'ycz_result.sta_re20190701';

% sql_str1 = ['select r1,r2 from ycz_result.sta_re20190701 where r1>0 and d =5 ',...
%     'and abs(r1)<0.1 and abs(closeprice/precoloseprice-1)<0.1 and abs(r2)<0.1'];

sql_str1 = ['select r2 from ycz_result.sta_re20190701 where r1>%0.2f and d =%d ',...
    'and abs(closeprice/precoloseprice-1)<0.1'];

t_all = 1:10;
r_all = 0.01:0.01:0.05;
m_x = zeros(10,length(r_all));
median_x = m_x;
sl_x = m_x;
sample_num = m_x;

for i = 1:length(t_all)
    sub_t = t_all(i);
    for j = 1:length(r_all)
        sub_r = r_all(j);
        sub_x= fetchmysql(sprintf(sql_str1,sub_r,sub_t));
        m_x(i,j) = mean(sub_x);
        median_x(i,j) = median(sub_x);        
        sample_num(i,j) = length(sub_x);
        sl_x(i,j)= sum(sub_x>0)/sample_num(i,j);
        
        sprintf('%d-%d',i,j)
        
    end
end
%按照文献方式排列
re = cell(5,1);
for i = 1:5
    re{i} = [m_x(:,i),median_x(:,i),sl_x(:,i),sample_num(:,i)]';
end


sql_str2= ['select r2 from ycz_result.sta_re20190701 where r1<-%0.2f and d =%d ',...
    'and abs(closeprice/precoloseprice-1)<0.1'];

m_x2 = m_x;
median_x2 = m_x;
sl_x2 = m_x;
sample_num2 = m_x;

for i = 1:length(t_all)
    sub_t = t_all(i);
    for j = 1:length(r_all)
        sub_r = r_all(j);
        sub_x= fetchmysql(sprintf(sql_str2,sub_r,sub_t));
        m_x2(i,j) = mean(sub_x);
        median_x2(i,j) = median(sub_x);        
        sample_num2(i,j) = length(sub_x);
        sl_x2(i,j)= sum(sub_x>0)/sample_num2(i,j);
        
        sprintf('%d-%d',i,j)
        
    end
end
%按照文献方式排列
re2 = cell(5,1);
for i = 1:5
    re2{i} = [m_x2(:,i),median_x2(:,i),sl_x2(:,i),sample_num2(:,i)]';
end

figure;
k = 1;
for i = 1:5
    ah=subplot(5,2,k);plot(ah,[m_x(:,i),median_x(:,i)]*100);
    ylabel(sprintf('%%%d',i))
    xlabel('分钟');
    axis(ah,'tight');
    k = k + 1;
    ah=subplot(5,2,k);plot(ah,[m_x2(:,i),median_x2(:,i)]*100);
    ylabel(sprintf('-%%%d',i))
    xlabel('分钟');
    axis(ah,'tight');
    k = k + 1;
end
legend({'平均值','中位值'})

