%% PairsTradingModel
% ��Խ���ģ��
% 1:��������--��ʼ�ʽ�V0,����feeH,feeA,stampH,stampA,��ȯ����sRateH,sRateA,�޷�������r,ʱ����dt,ѧϰʱ��N,Ŀ������T,�������ϵ��lambda,��Сkappa����minkappa
% 2:���ݴ���--����Ʊ�����ݴ�����ͨ������Լ����Э���Լ���
% 3:����ģ��--ÿ�ռ������ųֱֲ�����������ʵ�ʵĽ���ģ��
%   ����˵��������:
%       1:ʹ�õ������̼ۼ�������Ȩ�أ����赱�տ�����ɲ�λ������������ʹ�õ��տ��̼ۼ�������Ȩ�أ���������ɲ�λ������
%       2:���������ã���fee,��ƽfee,����fee+stamp+sRate,��ƽfee+stamp
clear;clc
% warning('off')
%% ���ݴ���
load JNJ;
load PG;

%stock = [JNJ PG];
check=1;%�Ƿ���м��飬1Ϊ����
fee=0;% ��������,0Ϊ�����Ƿ���
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
%% ����Ժ�Э���Լ���
if check==1
    % ����Լ���
    corHA=min(min(corrcoef(Nt,St)));
    if corHA<0.8
        disp([ '��Ʊ�������' num2str(floor(corHA*100)/100) '<0.8���������������'])
    end
    disp([ '��Ʊ�������' num2str(floor(corHA*100)/100) '>=0.8�������������'])
    % Э���Լ���(�ȼ�������Nt��St�Ƿ�ƽ�ȣ������ƽ�ȣ���ʹ�ò�ֽ���Э���Լ���)
    hH=adftest(Nt);
    hA=adftest(St);
    if hA*hH==0
        disp([ '�����ɼ۲�ƽ�ȣ�ʹ�ò�ֽ���Э���Լ���'])
        testData=[diff(Nt),diff(St)];
    else
        disp([ '�����ɼ�ƽ�ȣ�ʹ�ö����ɼ۽���Э���Լ���'])
        testData=[Dt,St];
    end
    [h,pValue,stat,cValue,reg1,reg2]=egcitest(testData);
    if h==0
        disp([ '��Ʊ�Բ�����Э����ϵ���������������'])
        %continue
    end
    disp([ '��Ʊ�Դ���Э����ϵ�������������'])
end

%% ��������
% �̶�����
V0=1000000;
r=0/100;
tDay=390;
cDay=390;
dt=1/tDay;
% �ɱ����    
[feeH,feeA,stampH,stampA,sRateH,sRateA]=feeControl(fee);% ��������
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

    %% ����ģ��
    %������������Ȩ��
    [position,paraSeries]=positionCaculate_new_sc_JNJ_PG(St,Xt,N,lambda,T,dt,tDay,minKappa);
    [accountDetail,tradeDetail]=tradeSimulate(position,priceH,priceA,dateS,V0,feeH,feeA,sRateH,sRateA,stampH,stampA,r,cDay);

    %% �������������
    result=resultRecord(accountDetail,tDay);        
    %resultMat(i,:)=[result,loopPara(i,:)];
    resultList{i,1}=accountDetail;
    resultList{i,2}=position;
    stockName = 'PG_JNJ';
    %% ��������ֱ�ۻ�ͼ
    plotShow(priceH,priceA,Xt,position,accountDetail,dateS,stockName,result);
    f=getframe(gcf);
    imwrite(f.cdata,['.\ͼƬ����\',testDate,'_',int2str(sampleNum), stockName,'.jpg']);

    %disp([stockName '��ɵ�' num2str(i) '���Ż�' ])
%end
sampleNum = 1;
testDate = today();
disp(['������ɵ�' num2str(sampleNum) '����Ʊ�ԡ���'  stockName])
eval(['save .\���ݱ���\' testDate '_result_' num2str(sampleNum) '.mat resultMat resultList'])

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

