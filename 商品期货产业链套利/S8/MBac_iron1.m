%{
ÂİÎÆ¸Ö RB
Ìú¿óÊ¯ I
½¹Ì¿ J

f_name_r{1}= containers.Map({'¶¹Ò»';'¶¹¶ş';'¶¹ÆÉ';'¶¹ÓÍ';'×ØéµÓÍ';'ÓñÃ×';'ÓñÃ×µí·Û';'¼¦µ°';'ÏËÎ¬°å';'½ººÏ°å';'¾ÛÒÒÏ©';'¾ÛÂÈÒÒÏ©';'¾Û±ûÏ©';'½¹Ì¿';'½¹Ãº';'Ìú¿óÊ¯';'ÒÒ¶ş´¼'},...
    {'A';'B';'M';'Y';'P';'C';'CS';'JD';'FB';'BB';'L';'V';'PP';'J';'JM';'I';'EG'});

%ÉÏÉÌËù¶ÔÓ¦Ãû³Æ
f_name_r{2}= containers.Map({'Í­';'ÂÁ';'Ğ¿';'Ç¦';'Äø';'Îı';'»Æ½ğ';'°×Òø';'ÂİÎÆ¸Ö';'Ïß²Ä';'ÈÈÔş¾í°å';'Ô­ÓÍ';'È¼ÁÏÓÍ';'Á¤Çà';'ÌìÈ»Ïğ½º';'Ö½½¬'},...
    {'CU';'AL';'ZN';'PB';'NI';'SN';'AU';'AG';'RB';'WR';'HC';'SC';'FU';'BU';'RU';'SP'});

%Ö£ÉÌËù¶ÔÓ¦Ãû³Æ
f_name_r{3} = containers.Map({'ÃŞ»¨','ÔçôÌ','²ËÓÍ','°×ÌÇ','PTA','Ç¿Âó','Ó²Âó','¼×´¼','²ËÓÍ','ÔçôÌ','Ç¿Âó','²£Á§','ÆÕÂó','ÓÍ²Ë×Ñ','²Ë×ÑÆÉ',...
    '¶¯Á¦Ãº','¶¯Á¦Ãº','¾¬µ¾','¼×´¼','ÍíôÌ','¹èÌú','ÃÌ¹è','ÃŞÉ´','Æ»¹û','PTA'},...
    {'CF','ER','RO','SR','PTA','WS','WT','ME','OI','RI','WH','FG','PM','RS','RM',...
    'TC','ZC','JR','MA','LR','SF','SM','CY','AP','TA'});
%}
clear

codes = {'RB','I','J'};
db_names = {'shfe','dce','dce'};
T = length(codes);
X = cell(T,1);
sql_str = 'select tradingdate,open,close from futuredata.price_if_data where variety=''%s'' order by tradingdate';
for i = 1:T
    sub_sql_str = sprintf(sql_str,codes{i});
    X{i} = fetchmysql(sub_sql_str,2);
end

[inds,tref] = suscc_intersect(X{1}(:,1),X{2}(:,1),X{3}(:,1));

P = [X{1}(inds(:,1),3),X{2}(inds(:,2),3),X{3}(inds(:,3),3)];
for i = 1:3
    X{i}=X{i}(inds(:,i),:);
end


P= cell2mat(P);
tref = datenum(tref);
ind = tref>=datenum(2013,10,18)&tref<=datenum(2016,12,30);
tref = tref(ind);
P = P(ind,:);
for i = 1:3
    X{i}=X{i}(ind,:);
end

d = P(:,1)-P(:,2)*1.6-P(:,3)*0.5;

%d1 = movavg(d,'simple',10);
d1 = zeros(size(d));
for i = 11:length(d1)
    d1(i) = mean(d(i-10:i-1));
end


%sd1 = movstd(d,[10,0]);
sd1 = zeros(size(d));
for i = 11:length(sd1)
    sd1(i) = std(d(1:i-1));
end
%get signal
T = length(d);
signal = zeros(T,1);
for i = 10:T
    %temp = d(i)>=d1(i)+sd1(i)&d(i)<=d1(i)+sd1(i)*1.2&d(i)<d(i-1)&d(i-1)<d(i-2);
    temp = d(i)>=d1(i)+sd1(i)&d(i)<=d1(i)+sd1(i)*1.2&d(i)<=d(i-1);
    if temp
        signal(i) = 1;
        continue
    end
        
    if d(i)<=d1(i)
        signal(i)=0;
        continue
    end
    
    signal(i) = signal(i-1);    
end

%bac
cash0 = 1000000;
para1 = [10,100,100];
i = 1;

x=cell2mat(X{1}(:,2:3));
x2=cell2mat(X{2}(:,2:3));
x3=cell2mat(X{3}(:,2:3));

v1=para1(1);
v2=para1(2);
v3=para1(3);
VB = zeros(T,1);
VU = VB;
vv = VB;

num = 3;
stop_v = -5/100;
vv(1) = cash0;
while i <T
    i = i + 1;
    if eq(signal(i-1),0)
        VB(i) = 0;
        VU(i) = 0;
    elseif eq(signal(i-1),-1)
        VB(i) = VB(i) + (x(i,2)-x(i,1))*v1*num;
        VU(i) = VU(i) - (x2(i,2)-x2(i,1))*v2*num- (x3(i,2)-x3(i,1))*v3*num;
    else
        VB(i) = VB(i) -(x(i,2)-x(i,1))*v1*num;
        VU(i) = VU(i) + (x2(i,2)-x2(i,1))*v2*num + (x3(i,2)-x3(i,1))*v3*num;
    end
    
    vv(i) = vv(i-1)+VB(i)+VU(i);
    if vv(i)/vv(i-1)-1<stop_v
        %vv(i:i+10) = vv(i-1)*(1+stop_v);
        vv(i:i+10) = vv(i);
        i = i + 10;        
    end

end



bpcure_plot_update(tref,vv);







