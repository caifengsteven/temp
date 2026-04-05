function [var_re,cvar_re,var_re1,cvar_re1] = var_cvar_norm(r,alpha)

[mx,stdx] = normfit(r);

var_re = mx-1.96*stdx;
cvar_re = mx-stdx*exp(-1.65*1.65/2)/sqrt(2*pi)/(1-alpha);

var_re1 = mx+1.96*stdx;
cvar_re1 = mx+stdx*exp(-1.65*1.65/2)/sqrt(2*pi)/(1-alpha);


