%M_hurst_exp
clear
cal_exp =@(n) (n-0.5)/n/sqrt(n*pi/2)*sum(sqrt((n-(1:n-1))./(1:n-1)));

T = 887;
y = zeros(T,1);
for i = 1:T
    y(i) = cal_exp(i);
end

subplot(1,2,1);plot(y);
set(gca,'YGrid','on');
title('R/S期望值和时间长度的关系');

subplot(1,2,2);
plot(1./(1:T));
title('R/S方差和样本观测数目关系')