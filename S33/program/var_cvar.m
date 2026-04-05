function [VaR,CVaR,VaR1,CVaR1] = var_cvar(r,alpha)
VaR = prctile(r,alpha * 100);
CVaR = (mean(r(r<=VaR)));


VaR1 = prctile(r,100-alpha * 100);
CVaR1 = (mean(r(r>=VaR1)));
