%指数数据
%EMA更换为MA
clear
%close all
t0 = datenum(2011,1,4);
tt = datenum(2019,12,30);

mod = 1;
%1 全部信号
%2 只做多
%3 只做空
warning('off');
sub_data_info = '中小板指';


vals = 1;
Y = [];
leg_str = [];

window1 = 22;
window2 = 22;

%获取数据
%temp = load(fn);
%x =[temp.x1.eob',num2cell([temp.x1.open;temp.x1.close]')];
x = fetchmysql(sprintf('select tradingdate,open,close from futuredata.indicator_data where symbolname = ''%s'' order by tradingdate',sub_data_info),2);
%sub_data_info = temp.x1.info;

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
%计算指标
%残差波动率
P_ma = MA(P,window1);
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
        if P(i)>P_ma(i)
            ind(i) = 1;
            %continue
        elseif P(i)<P_ma(i)
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
leg_str = cat(1,leg_str,{sub_data_info;[sub_data_info,'择时']});


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