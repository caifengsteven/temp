function x = stockNum2ID(x1)
%x1 = [1,3,600001];
[m,n] = size(x1);
if ~any(eq([m,n],1))
    %检查数据，输入需要为一列或者一行
    keyboard
end
if m<n
    x1 = x1';
end
testNum = floor(x1/100000);
ind1 = eq(testNum,3)|eq(testNum,0);
ind2 = eq(testNum,6);
ind3 = eq(testNum,9)|eq(testNum,2);
if ~eq(length(find((ind1+ind2+ind3)>0)),length(x1))
    keyboard
end
x = zeros(size(x1));
x(ind1) = x1(ind1)+22000000;
x(ind2) = x1(ind2)+11000000;
x(ind3) = x1(ind3)+33000000;
