function [CVaR1,CVaR2] = var_cvar_ref2(r,alpha)
    mx = mean(r);
    temp = Scaled_CVaR_Norm_Component(r , alpha );
    CVaR1 = mx-temp;
    CVaR2 = mx+temp;
end