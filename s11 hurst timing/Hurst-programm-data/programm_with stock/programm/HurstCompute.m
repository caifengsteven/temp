function HurstExponent=HurstCompute(Xtimes)
%HurstCompute
%code by ariszheng@gmail.com
%2008-10-07
%example HurstExponent=HurstCompute(rand(1,240))
%输入参数为Xtimes
LengthX=length(Xtimes);
%进行因式分解
[FactorMatrix,FactorNum]=HurstFactorization(LengthX);
%定义LogRS，为方便计算变量的初始一般为0
LogRS=zeros(FactorNum,1);
%定义LogN
LogN=zeros(FactorNum,1);
%分组计算
for i=1:FactorNum
    %根据因式分解方案，将数量进行分组
    %例如 FactorMatrix(i,:)=[8    30]
    %将240个元素的列向量，转换为8X30的矩阵
    dataM=reshape(Xtimes,FactorMatrix(i,:));
    %计算矩阵每列的均值
    MeanM=mean(dataM);
    %执行
    SubM =dataM - repmat( MeanM,FactorMatrix(i,1),1) ;
    RVector=zeros(FactorMatrix(i,2),1);
    SVector=zeros(FactorMatrix(i,2),1);
   %计算（R/S）n的累加
    for j=1:FactorMatrix(i,2)
        %SubVector=zeros(FactorMatrix(i,1),1);
        SubVector=cumsum( SubM(:,j));
        RVector(j)=max(SubVector)-min(SubVector);
        SVector(j)=std( dataM(:,j) );
    end
    %分别计算LogRS、LogN
    LogRS(i)=log( sum( RVector./SVector)/ FactorMatrix(i,2) );
    LogN(i)=log( FactorMatrix(i,1) );
end
%使用最小二乘法进行回归，计算赫斯特指数HurstExponent
HurstExponent=polyfit(LogN,LogRS,1);
