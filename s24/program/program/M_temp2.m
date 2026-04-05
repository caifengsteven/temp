%clear
load y1
T = size(y,1);
y2= zeros(T,2);

r_bond = exp(log(1.1)/244/6)-1;


for i = 1:T
    sub_y = y{i};
    if isempty(sub_y)
        continue
    end
    sub_y = sub_y';
    y2(i,:) = sub_y;
    
    
end
y2(:,1) = y2(:,1)+r_bond;

figure;
% subplot(2,1,2)
plot(cumprod(1+y2),'LineWidth',2)

y_c = cumprod(1+y2);
[v,v_str,sta_val] = curve_static0(y_c(:,1))
grid on