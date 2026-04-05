%M_static1
%均值 标准差 偏度 峰度 J-B检验 J-B检验临界值
clear
x = 1:10000;
r1 = sin(2*pi*x/1000);

hurst_rs(r1);
%hurst_rs_update1(r1,(3:50:5000)');
axis tight
%hurst_rs_test(r1);
