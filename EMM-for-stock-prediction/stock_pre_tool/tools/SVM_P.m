function [acc] =SVM_P(price,time)
%% ONLY WITH PRICE
%%%%%��ʼ��%%%%%%%%
fudu=bodong(price);
output=updown(price)';
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
output=output(time+1:count,1);
% EXACT DATA
trainX=input(1:fix(0.9*count),:);
trainY=output(1:fix(0.9*count),:);
testX=input(fix(0.9*count)+1:count-time,:);
testY=output(fix(0.9*count)+1:count-time,:);

model=svmtrain(trainY,trainX);
[predict_label, Accuracy, dec_values] =svmpredict(testY,testX,model);
acc=Accuracy(1,1);
end

