function hOptimal=OptimalCaculate_new_sc(sigma,mu,kappa,theta,eta,rho,lambda,T,t,x)
%% 函数说明
% 计算最优解h(t,x)
% 输入变量lambda为风险厌恶系数，T为周期以年为单位
%% 
alpha=kappa*(1-sqrt(1-lambda))/(2*eta^2)*(1+2*sqrt(1-lambda)/((1-sqrt(1-lambda))-(1+sqrt(1-lambda))*exp(2*kappa*(T-t)/sqrt(1-lambda))));
%beta=1/(2*eta^2*((1-sqrt(1-lambda))-(1+sqrt(1-lambda))*exp(2*kappa*(T-t)/sqrt(1-lambda))))*...
%    (lambda*sqrt(1-lambda)*(eta^2+2*rho*sigma*eta)*(1-exp(2*kappa*(T-t)/sqrt(1-lambda)))^2-...
%    lambda*(eta^2+2*rho*sigma*eta+2*kappa*theta)*(1-exp(2*kappa*(T-t)/sqrt(1-lambda))));
temp = exp(2*kappa*(T-t)/sqrt(1-lambda));
a = 1/(2*eta^2*((1-sqrt(1-lambda))-(1+sqrt(1-lambda))*exp(2*kappa*(T-t)/sqrt(1-lambda))));
b = lambda*sqrt(1-lambda)*(eta^2+2*rho*sigma*eta)*(1-exp(2*kappa*(T-t)/sqrt(1-lambda)))^2;
c =  lambda*(eta^2+2*rho*sigma*eta+2*kappa*theta)*(1-exp(2*kappa*(T-t)/sqrt(1-lambda)));
beta = a *(b-c);
beta=1/(2*eta^2*((1-sqrt(1-lambda))-(1+sqrt(1-lambda))*exp(2*kappa*(T-t)/sqrt(1-lambda))))*...
    (lambda*sqrt(1-lambda)*(eta^2+2*rho*sigma*eta)*(1-exp(2*kappa*(T-t)/sqrt(1-lambda)))^2-...
    lambda*(eta^2+2*rho*sigma*eta+2*kappa*theta)*(1-exp(2*kappa*(T-t)/sqrt(1-lambda))));

beta = kappa * theta/(eta^2)* (1+sqrt(1-lambda))*(exp(2*kappa/sqrt(1-lambda)*(T-t))-1)/(1+(1-2/(1-sqrt(1-lambda)))*exp(2*kappa/sqrt(1-lambda)*(T-t)));



hOptimal=1/(1-lambda)*(beta+2*x*alpha-kappa*(x-theta)/eta^2+rho*sigma/eta+1/2);

end