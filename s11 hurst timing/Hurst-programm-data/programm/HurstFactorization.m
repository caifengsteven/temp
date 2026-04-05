function  [FactorMatrix,FactorNum]=HurstFactorization(x)
%hurstFactorization
%code by ariszheng@gmail.com
%2008-10-07
%因子分解, 以4开始以X/4结束
%floor函数表示四舍五入
K = 4;

N=floor(x/K);
%方案数量初始为0
FactorNum=0;
%因子分解, 以4开始以X/4结束
for i=K:N
    %i可以被x整除,即得到一组分解方案
    if mod(x,i)==0
       %方案数量+1
        FactorNum=FactorNum+1;
        %将可行方案存储到FactorMatrix中
        FactorMatrix(FactorNum,:)=[i,x/i];
    end
end