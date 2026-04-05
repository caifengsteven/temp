%{
小波多尺度分析测试
%}
clear
%close all
%dwtmode('per')
load dataset1.mat
t1 = datenum(2002,1,1);
t2 = datenum(2016,6,30);
t3 = datenum(2014,1,1);
t4 = datenum(2005,2,1);

%图3 日收益率相关性
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

T0 = 7;
ind = find(tref_num>=t4,1);
ind = ind:(ind+1024-1);
tref = tref(ind);
sub_y = yield_v(ind,:);
%sub_y = log(x(ind,:));
%sub_y = x(ind,:);
[h,pValue,stat,cValue,reg1,reg2] = egcitest(sub_y);
sub_y1 = reg1.res;
[A_a,D_a] = wt_msr(sub_y1',T0,'db8',0);


for i = 1:T0
    subplot(T0,1,T0-i+1);plot(D_a{i});
    set(gca,'xlim',[0,length(sub_y1)+1]);
    ylabel(sprintf('db%d',i))
end
y =  movmax(abs(D_a{1}),[100,0]);
figure;
bar(y);






