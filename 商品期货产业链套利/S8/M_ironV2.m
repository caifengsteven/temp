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
tref = tref(tref>=datenum(2014,07,28)&tref<=datenum(2016,12,30));
T = length(tref);
num = 2;
sql_str2 = 'select close from futuredata.price_%s_data where tradingdate=''%s'' and variety=''%s'' order by codename';
X = zeros(T,3);
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
        X(i,j) = temp(num);
    end
    i
end
P = X;
v2 = P(:,1)-P(:,2)*1.6-P(:,3)*0.5-800;
plot(tref,v2,'LineWidth',2)
datetick('x','yymm')