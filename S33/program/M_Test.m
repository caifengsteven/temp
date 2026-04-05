%M_Test
clear
sql_str = 'select close from ycz_min_history.`20100104` where symbol = ''sh600000'' order by tradingdate';
x = fetchmysql(sql_str);
r = x(2:end)./x(1:end-1)-1;

alpha = 0.05;

%[VaR,CVaR,VaR1,CVaR1] = var_cvar(r,alpha);
%[VaR,CVaR,VaR1,CVaR1] = var_cvar_test(r,0.95);
%[VaR,CVaR,VaR1,CVaR1] = var_cvar_norm(r,0.95);

[CVaR,CVaR1] = var_cvar_ref2(r,0.95);
histogram(r)
lims = axis(gca);
hold on

%plot([VaR,VaR],[lims(3),lims(4)],'r--')

%plot([VaR1,VaR1],[lims(3),lims(4)],'r--')

plot([CVaR,CVaR],[lims(3),lims(4)],'k--')

plot([CVaR1,CVaR1],[lims(3),lims(4)],'k--')