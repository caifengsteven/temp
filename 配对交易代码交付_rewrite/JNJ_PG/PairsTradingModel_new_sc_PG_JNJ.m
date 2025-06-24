%% PairsTradingModel
% 配对交易模型
% 1:参数设置--初始资金V0,费率feeH,feeA,stampH,stampA,融券利率sRateH,sRateA,无风险利率r,时间间隔dt,学习时间N,目标周期T,风险厌恶系数lambda,最小kappa设置minkappa
% 2:数据处理--将股票对数据处理，并通过相关性检验和协整性检验
% 3:交易模拟--每日计算最优持仓比例，并进行实际的交易模拟
%   交易说明及假设:
%       1:使用当日收盘价计算最优权重，假设当日可以完成仓位调整（报告中使用当日开盘价计算最优权重，并当日完成仓位调整）
%       2:手续费设置：买开fee,买平fee,卖开fee+stamp+sRate,卖平fee+stamp
clear;clc
% warning('off')
%% 数据处理
load JNJ;
load PG;

%stock = [JNJ PG];
check=1;%是否进行检验，1为检验
fee=0;% 费率设置,0为不考虑费率
resultList{1,1}=[];
ind=1;
tic

%dateS=stockData{sampleNum,1}(:,1);
dateS = table2array(PG(:,1));
%dateS = datetime(dateS,'InputFormat','yyyy-MM-dd HH:mm:ss');
priceH=table2array(PG(:,3));
priceA=table2array(JNJ(:,3));
St=log(priceA);
Nt=log(priceH);
Xt=log(priceH)-log(priceA);
%St = log(priceH);
%Nt = log(priceA);
%Xt = log(priceA) - log(priceH);
%% 相关性和协整性检验
if check==1
    % 相关性检验
    corHA=min(min(corrcoef(Nt,St)));
    if corHA<0.8
        disp([ '股票对相关性' num2str(floor(corHA*100)/100) '<0.8，不满足配对条件'])
    end
    disp([ '股票对相关性' num2str(floor(corHA*100)/100) '>=0.8，满足配对条件'])
    % 协整性检验(先检验序列Nt与St是否平稳，如果不平稳，则使用差分进行协整性检验)
    hH=adftest(Nt);
    hA=adftest(St);
    if hA*hH==0
        disp([ '对数股价不平稳，使用差分进行协整性检验'])
        testData=[diff(Nt),diff(St)];
    else
        disp([ '对数股价平稳，使用对数股价进行协整性检验'])
        testData=[Dt,St];
    end
    [h,pValue,stat,cValue,reg1,reg2]=egcitest(testData);
    if h==0
        disp([ '股票对不存在协整关系，不满足配对条件'])
        %continue
    end
    disp([ '股票对存在协整关系，满足配对条件'])
end

%% 参数设置
% 固定参数
V0=1000000;
r=0/100;
tDay=390;
cDay=390;
dt=1/tDay;
% 可变参数    
[feeH,feeA,stampH,stampA,sRateH,sRateA]=feeControl(fee);% 费率设置
sRateH = 0;
sRateA = 0;

N=120;
lambda=-300;
T=1;
minKappa=-40;
%load loopPara
%for i=1:size(loopPara,1)
%    N=loopPara(i,1);
%    lambda=loopPara(i,2);
%    T=loopPara(i,3);
%    minKappa=loopPara(i,4);

    %% 交易模拟
    %计算最优配置权重
    [position,paraSeries]=positionCaculate_new_sc_JNJ_PG(St,Xt,N,lambda,T,dt,tDay,minKappa);
    [accountDetail,tradeDetail]=tradeSimulate(position,priceH,priceA,dateS,V0,feeH,feeA,sRateH,sRateA,stampH,stampA,r,cDay);

    %% 结果分析及保存
    result=resultRecord(accountDetail,tDay);        
    %resultMat(i,:)=[result,loopPara(i,:)];
    resultList{i,1}=accountDetail;
    resultList{i,2}=position;
    stockName = 'PG_JNJ';
    %% 单个测试直观画图
    plotShow(priceH,priceA,Xt,position,accountDetail,dateS,stockName,result);
    f=getframe(gcf);
    imwrite(f.cdata,['.\图片保存\',testDate,'_',int2str(sampleNum), stockName,'.jpg']);

    %disp([stockName '完成第' num2str(i) '个优化' ])
%end
sampleNum = 1;
testDate = today();
disp(['计算完成第' num2str(sampleNum) '个股票对——'  stockName])
eval(['save .\数据保存\' testDate '_result_' num2str(sampleNum) '.mat resultMat resultList'])

tradeStockSharp{ind,1}=sampleNum;
tradeStockSharp{ind,2}=stockName;
[~,index1]=sort(resultMat(:,4),'descend');
tradeStockSharp{ind,3}=resultMat(index1(1),:);
tradeStockSharp{ind,4}=resultList{index1(1),1};
tradeStockSharp{ind,5}=resultList{index1(1),2};

tradeStockCammar{ind,1}=sampleNum;
tradeStockCammar{ind,2}=stockName;
[~,index2]=sort(resultMat(:,2)./resultMat(:,8),'descend');
tradeStockCammar{ind,3}=resultMat(index2(1),:);
tradeStockCammar{ind,4}=resultList{index2(1),1};
tradeStockCammar{ind,5}=resultList{index1(1),2};

ind=ind+1;
toc

eval(['save tradeStock_' testDate ' tradeStockSharp tradeStockCammar'])

