%{
小波多尺度分析测试
突变点
limit ind
[89,212,321,403,459,833,887,929,415,992,1014,1067,1117,1136,1166,1186,1208,1252,...
1323,1412]
global ind
[2585;2708;2817;2899;2955;3329;3383;3425;2911;3488;3510;3563;3613;3632;3662;...
3682;3704;3748;3819;3908]

%}
clear
%close all
%dwtmode('per')
load dataset3.mat

%t1 = datenum(2009,12,31);
%t2 = datenum(2016,6,30);

%图3 日收益率相关性
yield_v = zeros(size(x));
yield_v(2:end,:) = x(2:end,:)./x(1:end-1,:)-1;

T0 = 7;
ind = 1:length(tref);
%ind = find(tref_num>=t1&tref_num<=t2);
%ind = ind:(ind+1024-1);
tref = tref(ind);
%sub_y = yield_v(ind,:);
sub_y = log(x(ind,:));
%sub_y = x(ind,:);
[h,pValue,stat,cValue,reg1,reg2] = egcitest(sub_y);
sub_y1 = reg1.res;
[A_a,D_a] = wt_msr(sub_y1',T0,'db8',0);


for i = 1:T0
    subplot(T0+1,1,T0-i+1);plot(D_a{i});
    set(gca,'xlim',[0,length(sub_y1)+1]);
    ylabel(sprintf('db%d',i))
end
y =  movmax(abs(D_a{1}),[20,0]);
subplot(T0+1,1,T0+1)
bar(y);






