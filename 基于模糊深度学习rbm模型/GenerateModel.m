%RBMѵ���õ���һ��������������rbm����Ϊ����
rl = rbm([30, 100]);
r1 = checkrbmtrain (@rbmtrainl, rbml, Xja5000, 50, 0.1);
net_rl = rbm2nnet (rbml,'up');
hl = nnetfw (net_r1, XJA5000);
%RBMѵ���õ��ڶ�������������������Ϊ��һ��������
r2=rbm([100, 80]);
r2=checkrbmtrain (@rbmtrainl, rbm2, h1, 50, 0.1);
net_rbm2=rbm2nnet(r2,'up');
h2=nnetfw(net,rbm2,h1);
%RBMѵ���õ�����������������������Ϊ�ڶ���������
r3 = rbm([80, 60])
r3=checkrbmtrain(@rbmtrain1,rbm3, h2, 50,0.1);
net_rbm3=rbm2nnet(rbm3,'up');
h3=nnetfw(net_rbm3, h2);
%RBMѵ���õ�����������������������Ϊ������������
r4 = rbm([60,20]);
r4=checkrbmtrain(@rbmtrainl, rbm4, h3, 50, 0.l);
net_rbm4 = rbm2nnet(rbm4,'up');
h4= nnetfw (net_rbm4, h3);
%RBMѵ���õ�����������������������Ϊ������������
r5=rbm([20, 10]);
r5 = checkrbmtrain(@rbmtrainl, rbm5, h4, 50,0.l);
net_rbm5 = rbm2nnet (rbm5,'up');
%����������磬����ʼ������Ϊѵ�����Ĳ�����
h5 = nnetfw(net_rbm5,h4);
net1 = nnet([6, 100, 80, 60, 20, 10, 3],'softmax')
net1.w{1} = net_rbml.w{1};
net1.w{2} = net_rbm2.w{1};
net1.w{3} = net_rbm3.w{1};
net1.w{4} = net_rbm4.w{1};
net1.w{5} = net_rbm5.w{1};

%������������BPѵ��

net2=nnettrain (netl, XJA5000, Y, 1000)