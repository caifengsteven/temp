function [acc] = BPnet(VarName5,time,node,epochs )
%%%%%%%%%%VarName1 Ϊ��Ʊ�Ŀ���ʱ�� VarName5Ϊ��Ӧ�Ĺ�Ʊ�۸�timeΪ��ʷʱ�䴰���ȣ�nodeΪ�������������ڵ���
price=VarName5;%%��Ʊ�۸�
fudu=bodong(price);
%% ONLY WITH PRICE
%%%%%��ʼ��%%%%%%%%
shuju=fudu;
[count,factors]=size(shuju);
[shuju]=p_guiyihua(shuju);
%%%%%%%%%%��Ϊtimeά������%%%%%%
input=shuju(1:count-time+1,:);
for i=2:time
    temp=shuju(i:count-time+i,:);
    input=[input,temp];
end
input=input(1:count-time,:);
output=shuju(time+1:count,1);
% EXACT DATA
trainX=input(1:fix(0.9*count),:);
trainY=output(1:fix(0.9*count),:);
testX=input(fix(0.9*count)+1:count-time,:);
testY=output(fix(0.9*count)+1:count-time,:);
factors=1;

net=newff(minmax(trainX'),[node factors],{'logsig','logsig'},'traingdx','learngdm');

%%����ѵ������
net.trainFcn='trainlm';
net.trainparam.show=50;
net.trainparam.epochs=epochs;
net.trainparam.goal=0.000003;
net.trainparam.lr=0.02;
%��ʼѵ��
net=train(net,trainX',trainY');
%%����
y=sim(net,testX');
[s1 s2]=size(y);
temp=y;
y(find(y>0.5))=1;
y(find(y<=0.5))=0;
testY(find(testY>0.5))=1;
testY(find(testY<=0.5))=0;
%%
right=zeros(length(y),1);
for i=1:length(y)
    if testY(i)==y(i)
        right(i)=1;
    end
end
disp(' ');
acc=sum(right)/length(y);


end

