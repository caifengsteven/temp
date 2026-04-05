%{
谱分析确定窗口
%}
function [y_sig2,fre_sel] = cal_freq(y,N_pool,window)
N_pool(N_pool>size(y,1)) = [];
    if nargin < 3
        window = max(N_pool);
    end
    y_sig2 = zeros(length(y),length(N_pool));
    for i = 1:length(N_pool)
        temp_mean = movmean(y,[N_pool(i)-1,0]);
        temp_mean(1:window-1)=0;
        for j = window:length(y)        
            y_sig2(j,i) = std(y(j-N_pool(i)+1:j)-temp_mean(j));
        end
    end
    v = std(y_sig2(window:end,:));
    [~,ia] = min(v);
    fre_sel = N_pool(ia);
end