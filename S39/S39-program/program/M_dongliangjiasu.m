clear
x = xlsread('data.xlsx');

p_window = 18;
x1 = x(:,1);
T = length(x1);
y1 = zeros(T,1); %
y2 = y1;
for i = p_window:T
    sub_wid = i-p_window+1:i;
    sub_x = x1(sub_wid);
    
    temp = cumprod((1+sub_x).^(1/p_window))-1;
    y1(i) = temp(end);
    
    temp2 = polyfit((1:p_window)',sub_x,2);
    y2(i) = temp2(1);
end


function get_