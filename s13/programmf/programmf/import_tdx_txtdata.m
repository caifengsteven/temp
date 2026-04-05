function [x,etf_name,etf_code] = import_tdx_txtdata(fn)
fid = fopen(fn);
x = textscan(fid,['%s ',repmat('%f ',1,18)],'HeaderLines',3);
fclose(fid);
x = [x{1},num2cell([x{2},x{5}])];
x = x(1:end-1,:);

[~,fn] = fileparts(fn);
n = strfind(fn,'ETF');
etf_name = fn(1:n+2);
etf_code = fn(n+3:end-4);
end