function [guiyihua] = guiyihua(shuju)
M=max(shuju);
m=min(shuju);
[n,k]=size(shuju);
A=zeros(n,k);
B=zeros(n,k);
for i=1:n
  A(i,:)=m;%biΪ��֪��������
  B(i,:)=M-m;
end

guiyihua=(shuju-A)./B;
end
