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

P= cell2mat(P);
tref = datenum(tref);
ind = tref>=datenum(2013,10,18)&tref<=datenum(2016,11,30);
tref = tref(ind);
P = P(ind,:);

plot(tref,P(:,1)-P(:,2)*1.6-P(:,3)*0.5,'LineWidth',2)
datetick('x','yymm')