%ָ������

clear
close all
t0 = datenum(2011,1,4);
tt = datenum(2018,12,30);

mod = 1;
%1 ȫ���ź�
%2 ֻ����
%3 ֻ����

fn = 'HSCEI_1';
vals = 2;
Y = [];
leg_str = [];
%for sel = 1:2
window1 = 20;
window2 = 5;

%��ȡ����
temp = load(fn);
x =[temp.x1.eob',num2cell([temp.x1.open;temp.x1.close]')];
sub_data_info = temp.x1.info;

%time
tref_str = x(:,1);
tref = datenum(x(:,1));

ind = tref>=t0&tref<=tt;
tref_str = tref_str(ind);
tref = tref(ind);
x = x(ind,:);

%price
P_o = cell2mat(x(:,2));
P = cell2mat(x(:,3));
%����ָ��
%�в����
P_ema = EMA(P,window1);
T = length(P);
RVol = zeros(T,1);
for i = window1:T
    sub_wid = i-window1+1:i;
    %sub_wid = 1:i;
    RVol(i) = std(P(sub_wid)-P_ema(sub_wid))*sqrt(window1);
end

RVol_ema = EMA(RVol,window2);
RVol_std = movstd(RVol-RVol_ema,[window2,0]);
ind = zeros(T,1);

for i = max(window1,window2)+1:T-1
    if RVol(i)-RVol_ema(i)>RVol_std(i)*vals
        if P(i)>P_ema(i)
            ind(i) = 1;
            %continue
        elseif P(i)<P_ema(i)
            ind(i) = -1;
            %continue
        else
            ind(i) = 0;
            %continue
        end
    else
        ind(i) = 0;
        %continue
    end
    ind(i+1) = ind(i);    
end

r = zeros(size(P));
r(2:end) = P(2:end)./P(1:end-1)-1;

ind2 = ind;
ind2(2:end) =ind(1:end-1);
ind2(1) = 0;
if eq(mod,2)
    ind2(ind2<0) = 0;
elseif eq(mod,3)
    ind2(ind2>0) = 0;
end
r1 = r.*ind2;

Y = cat(2,Y,cumprod(1+[r,r1]));
leg_str = cat(1,leg_str,{sub_data_info;[sub_data_info,'��ʱ']});

%end

plot(tref,Y,'LineWidth',2);
ah = gca;
ah.XTickLabelRotation=30;
datetick('x','yyyymmdd');
legend(leg_str)
re = [];
for i = 1:size(Y,2)
    [v,v_str,sta_val] = curve_static(Y(:,i));
    re = cat(2,re,v');
end