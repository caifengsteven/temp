%% 生成参数优化数组并保存
clear;clc
i=1;
loopPara(1,3)=0;
for N=[30,50:50:300]
    for lambda=-50:-50:-300
        for T=0.5:0.5:3
            %for minKappa=[-1000,10:10:50]
            for minKappa=[-1000]
                loopPara(i,1:4)=[N,lambda,T,minKappa];
                i=i+1;
            end
        end
    end
end
save loopPara loopPara