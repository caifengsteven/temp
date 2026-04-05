%clear
load y0
T = size(y,1);
y2= zeros(T,2);
for i = 1:T
    sub_y = y{i};
    if isempty(sub_y)
        continue
    end
    sub_y = sub_y';
    %del_ind = isnan(sum(sub_y,2));
    %sub_y(del_ind,:) = [];
    y2(i,:) = mean(sub_y);
    
    
end

% figure
% subplot(2,1,1)
% plot(cumprod(1+y2),'LineWidth',2)


y3= zeros(T,2);
for i = 1:T
    sub_y = y{i};
    if isempty(sub_y)
        continue
    end
    sub_y = sub_y';
    
    del_ind = eq(sub_y(:,1),0);
    
    sub_y1 = sub_y(:,1);
    sub_y1 = sub_y1(~del_ind);
    sub_y1(isnan(sub_y1)) = [];
    temp_y2 = zeros(1,2);
    if ~isempty(sub_y1)
        temp_y2(1) = mean(sub_y1);
    end
    
    sub_y2 = sub_y(:,2);
    sub_y2 = sub_y2(~del_ind);
    if ~isempty(sub_y2)
        temp_y2(2) = mean(sub_y2);
    end
    y3(i,:) = temp_y2;
end
figure;
% subplot(2,1,2)
axes( 'YAxisLocation', 'right', 'XAxisLocation', 'top')

r_bond = exp(log(1.1)/244/6)-1;

y3(:,1) = y3(:,1)+r_bond;



plot(cumprod(1+y3),'LineWidth',2)

grid on
y_c = cumprod(1+y3);
[v,v_str,sta_val] = curve_static0(y_c(:,1))