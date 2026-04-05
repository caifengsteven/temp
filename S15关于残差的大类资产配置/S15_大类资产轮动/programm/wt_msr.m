%wavelet multi-scale analysis
%输入数据 每行代表一个信号
function [A_a,D_a] = wt_msr(X,N,filter_name,disp_sel)

if nargin < 3
    filter_name = 'db2';
end
if isempty(filter_name)
    filter_name = 'db2';
end

if nargin<4
    disp_sel = 1;
end
if isempty(disp_sel)
    disp_sel = 1;
end

[m1,n1] = size(X);
X = [fliplr(X),X,fliplr(X)];
[m,~] = size(X);

A = cell(m,N);
D = A;
for i = 1:m
    s = X(i,:);
    [c,l] = wavedec(s,N,filter_name);
    for j = 1:N
        A{i,j} = wrcoef('a',c,l,filter_name,j)';
        D{i,j} = wrcoef('d',c,l,filter_name,j)';
    end
end
A_a = cell(N,1);
D_a = A_a;
for i = 1:N
    temp = A(:,i);
    A_a{i} = [temp{:}];
    A_a{i}=A_a{i}(n1+1:n1*2,:);
    temp = D(:,i);
    D_a{i} = [temp{:}];
    D_a{i}=D_a{i}(n1+1:n1*2,:);
end

if eq(disp_sel,1)
    figure;
    for i = 1:N
        subplot(N,2,i*2-1)
        plot(A_a{i});
        subplot(N,2,i*2)
        plot(D_a{i});
    end
end