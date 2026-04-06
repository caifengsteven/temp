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
»ñÈ¡Á¬1Á¬3Á¬9Êı¾İ
num=1+Á¬Êı

%}
clear

codes = {'RB','I','J'};
tb_names = {'shfe','dce','dce'};
sql_str1 = 'select distinct tradingdate from futuredata.price_shfe_data where variety=''RB'' order by tradingdate';
tref = fetchmysql(sql_str1,2);

tref = datenum(tref);
tref = tref(tref>=datenum(2013,10,18)&tref<=datenum(2016,12,30));
T = length(tref);
num = 2;
sql_str2 = 'select open,close from futuredata.price_%s_data where tradingdate=''%s'' and variety=''%s'' order by codename';
P = zeros(T,3);
X = P;
parfor i = 1:T
    sub_vec = datevec(tref(i));
    sub_t = datestr(tref(i),'yyyy-mm-dd');
    sub_strs = cell(3,1);
    sub_strs{1}=datestr(datenum(sub_vec(1),sub_vec(2)+1,1),'yymm');
    sub_strs{2}=datestr(datenum(sub_vec(1),sub_vec(2)+3,1),'yymm');
    sub_strs{3}=datestr(datenum(sub_vec(1),sub_vec(2)+9,1),'yymm');
    for j = 1:3
        sub_str = sprintf(sql_str2,tb_names{j},sub_t,codes{j});
        temp = fetchmysql(sub_str)
        P(i,j) = temp(num,2);
        X(i,j) = temp(num,1);
    end
    i
end
d = P(:,1)-P(:,2)*1.6-P(:,3)*0.5-1100;

d1 = movavg(d,'simple',10);
sd1 = movstd(d,[10,0]);

%get signal
T = length(d);
signal = zeros(T,1);
for i = 10:T
    %temp = d(i)>=d1(i)+sd1(i)&d(i)<=d1(i)+sd1(i)*1.2&d(i)<d(i-1)&d(i-1)<d(i-2);
    temp = d(i)>=d1(i)+sd1(i)&d(i)<=d1(i)+sd1(i)*1.2&d(i-1)<d(i-2);
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

x=[X(:,1),P(:,1)];
x2=[X(:,2),P(:,2)];
x3=[X(:,3),P(:,3)];

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
%     if vv(i)/vv(i-1)-1<stop_v
%         vv(i:i+10) = vv(i-1)*(1+stop_v);
%         %vv(i:i+10) = vv(i);
%         i = i + 10;        
%     end

end



bpcure_plot_update(tref,vv);












