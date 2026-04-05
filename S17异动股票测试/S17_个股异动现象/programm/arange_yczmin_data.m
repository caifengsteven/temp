function x_new = arange_yczmin_data(x)
s = unique(x(:,1));
T = length(s);
x_new = cell(T,3);
ind = zeros(T,1);
for i = 1:length(s)
    ia = strcmp(x(:,1),s(i));
    sub_x = x(ia,:);
    if eq(size(sub_x,1),2)
        x_new(i,:) = [sub_x(1,1),sub_x(1,2),sub_x(2,3)];
    else
        ind(i) = 1;
    end
    
    
end
x_new(eq(ind,1),:) = [];