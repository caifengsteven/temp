%{
导入数据
黄金缺少{'2001/9/11';'2006/1/2';'2006/11/24';'2006/7/3';'2014/5/25';'2014/8/31';'2016/12/25';'2016/3/25';'2017/4/14';'2019/4/19'}
oil缺少{'2001/6/4';'2002/11/26';'2002/3/27';'2003/11/25';'2004/5/27';'2016/5/29'}
%}
clear
[~,~,data_gold] = xlsread('黄金期货历史数据2.xlsx');
[~,~,data_oil] = xlsread('WTI原油期货历史数据.xlsx');
data_gold = data_gold(2:end,1:2);
data_oil = data_oil(2:end,1:2);
[~,ia,ib] = intersect(data_gold(:,1),data_oil(:,1));
data1 = [data_gold(ia,:),data_oil(ib,2)];
tref_num = datenum(data1(:,1));
[tref_num,ia] = sort(tref_num);
data1 = data1(ia,:);
tref = data1(:,1);
x = cell2mat(data1(:,2:3));
save dataset1 tref tref_num x