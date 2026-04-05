function [YF,YMSE]  = ar_pred(r0,wid)
Mdl = arima(12,0,0); 
T = length(r0);
YF = zeros(T,1);
YMSE = YF;

parfor  i = wid:T
    %sub_wid = 1:i;
    sub_wid = i-wid+1:i;
    EstMdl  = estimate(Mdl,r0(sub_wid),'Display','off');
    [YF(i),YMSE(i)] = forecast(EstMdl,1,'Y0',r0(sub_wid));
end

end