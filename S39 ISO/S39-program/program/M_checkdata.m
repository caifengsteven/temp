clear
[~,~,x1]=xlsread('return1.csv');
[~,~,x2] = xlsread('return_adair.csv');
x2(:,1) = [];
y = [x1(:,end-3:end),x2(:,end-3:end)];
y_t = y(1,:);
y = cell2mat(y(2:end,:));
x1 = x1(:,1:end-4);
x2 = x2(:,1:end-4);
x1 = cell2mat(x1);
x2 = cell2mat(x2);
[~,ia,ib] = intersect(x1(1,:),x2(1,:));
x1 = x1(:,ia);
x2 = x2(:,ib);
symbol = x1(1,:);
x1 = x1(2:end,:);
x2 = x2(2:end,:);
