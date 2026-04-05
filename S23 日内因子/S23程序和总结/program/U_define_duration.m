%计算时间间隔
%借用了datetime函数
function t0 = U_define_duration(t,N)

M = calmonths(N);
%datenum to datetime
t = datetime(datevec(t));
t0 = t-M;
t0 = datenum(t0);

