clear
fn = 'dataset3.xlsx';
fn_out = 'dataset3';
[~,~,x] = xlsread(fn);

x = x(2:end,:);
tref = x(:,1);
tref_num = datenum(tref);
x = cell2mat(x(:,2:end));

save(fn_out,'x','tref','tref_num')