%% 输出策略结果并保存到当前目录下
clear;clc;
load stockData;
testDate='all';%'less'
eval(['load tradeStock_' testDate ';']);
caName{1,1}='Sharp';
caName{2,1}='Cammar';
for n=1:2
    eval(['tradeStock=tradeStock' caName{n,1} ';']);
    eval([caName{n,1} 'outputTable{1,8}=[];'])
    for i=1:size(tradeStock,1)
        result=tradeStock{i,3}(1:10);
        loopPara=tradeStock{i,3}(11:14);
        eval([caName{n,1} 'outputTable(i,1:2)=tradeStock(i,1:2);'])
        eval([caName{n,1} 'outputTable(i,3:8)=num2cell(result(1,[1,2,4,8,9,10]));'])
        eval([caName{n,1} 'outputTable(i,9:12)=num2cell(loopPara(1,:));'])

        sampleNum=tradeStock{i,1};
        stockName=tradeStock{i,2}; 
        priceH=stockData{sampleNum,1}(:,2);
        priceA=stockData{sampleNum,1}(:,3);
        dateS=stockData{sampleNum,1}(:,1);    
        Xt=log(priceH)-log(priceA);
        accountDetail=tradeStock{i,4};
        position=tradeStock{i,5};
        close;
        plotShow(priceH,priceA,Xt,position,accountDetail,dateS,stockName,result);
        f=getframe(gcf);
        imwrite(f.cdata,['.\图片保存\',testDate,'_',caName{n,1},'_',int2str(sampleNum), stockName,'.jpg']);
    end
end
