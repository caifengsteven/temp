function [position,paraSeries]=positionCaculate_new_sc_JNJ_PG(St,Xt,N,lambda,T,dt,tDay,minKappa)
%% 函数说明
% 根据最优控制理论计算最优持仓权重
% |持仓比例|<=1
% paraSeries:原始仓位/限制仓位/sigma/mu/kappa/theta/eta/rho
%%
totalT=length(Xt);
paraSeries(totalT,8)=0;
for t=N+1:totalT
    StCal=St(t-N:t);
    XtCal=Xt(t-N:t);
    x=XtCal(end-1);%x取值有待考证
    %x=XtCal(N);%简化模型
    %x=0;
    %[sigma,mu,kappa,theta,eta,rho]=ParaEstimate(StCal,XtCal,N,dt,minKappa);   
    [sigma,mu,kappa,theta,eta,rho]=ParaEstimate_new_sc_JNJ_PG(StCal,XtCal,N,dt,minKappa);    
    setT=mod(t,T*tDay)/(T*tDay);
    %setT = t
%     setT=0;%简化模型
    hOptimal=OptimalCaculate_new_sc(sigma,mu,kappa,theta,eta,rho,lambda,T,setT,x);
    paraSeries(t,1)=hOptimal;
    hOptimal=min(max(hOptimal,-1),1);
    paraSeries(t,2)=hOptimal;
    paraSeries(t,3:8)=[sigma,mu,kappa,theta,eta,rho];
end
position=paraSeries(:,2);
end
