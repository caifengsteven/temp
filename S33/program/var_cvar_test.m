function [var_re,cvar_re,var_re1,cvar_re1] = var_cvar_test(r,alpha)
r_alpha = 1-alpha;
n = length(r);
mx = mean(r);
stdx = std(r);
t_ref = tinv(1-r_alpha/2,n-1);
var_re = mx-t_ref*stdx/sqrt(n);
cvar_re = mx-stdx/sqrt(n)*exp(-1.65*1.65/2)/sqrt(2*pi)/(1-alpha);

var_re1 = mx+t_ref*stdx/sqrt(n);
cvar_re1 = mx+stdx/sqrt(n)*exp(-1.65*1.65/2)/sqrt(2*pi)/(1-alpha);


