clear



dos('python bac_toolS105.py');
%[~,~,x] = xlsread(fullfile('计算结果','bac_curve_S102.csv'));
[~,~,x] = xlsread('S105_bac.xlsx');
tref = cellstr(datestr(datenum(x(2:end,1)),'yyyymmdd'));
X = cell2mat(x(2:end,2:end));
T = size(X,2);

H = zeros(T,1);
Y = cell(T,1);
%info = x(1,2:end);
info = {'S105-curve'};
ind = zeros(T,1);
for i = 1:T
    y_re = cumprod(1+X(:,i));
    H(i) = bacFigure(y_re,tref,info{i},[]);
    Y{i} = y_re;
end
report_adair('S105计算结果',H,Y,info);