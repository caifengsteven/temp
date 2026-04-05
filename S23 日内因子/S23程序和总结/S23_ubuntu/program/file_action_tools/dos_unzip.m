function dos_unzip(fn,pn)
if ~exist(pn,'dir')
    mkdir(pn)
end
%order_str = sprintf('"C:\\Program Files\\WinRAR\\WinRAR" x %s %s -O+',fn,pn); %覆盖已经存在的文件
order_str = sprintf('"C:\\Program Files\\WinRAR\\WinRAR" x %s %s -O-',fn,pn);  %不要覆盖已经存在的文件

dos(order_str)
