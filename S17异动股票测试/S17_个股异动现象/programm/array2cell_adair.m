function Y = array2cell_adair(y)
n = size(y,2);
Y = cell(1,n);
for i = 1:n
    Y{i} = y(:,i);
end