clear
fn = '交银180治理ETF510010.txt';
fid = fopen(fn);
x = textscan(fid,['%s ',repmat('%f ',1,18)],'HeaderLines',3);
fclose(fid);
x = [x{1},num2cell([x{2},x{5}])];
x = x(1:end-1,:);
n = strfind(fn,'ETF');
etf_name = fn(1:n+2);
etf_code = fn(n+3:end-4);
