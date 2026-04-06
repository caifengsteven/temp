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
Á¬1 Ò»ÔÂ·İºÏÔ¼Á¬ÆğÀ´µÄÒâË¼

%}
clear

key1='10';

v1 = num2cell(12:17);
v2 = cellfun(@(x) ['RB',num2str(x),key1],v1,'UniformOutput',false);
v2_str = ['''',strjoin(v2,''','''),''''];

sql_str = 'select tradingdate,close from futuredata.price_shfe_data where codename in(%s) order by tradingdate';
x_RB = fetchmysql(sprintf(sql_str,v2_str),2);

v2_I = cellfun(@(x) ['I',num2str(x),key1],v1,'UniformOutput',false);
v2_str = ['''',strjoin(v2_I,''','''),''''];
sql_str = 'select tradingdate,close from futuredata.price_dce_data where codename in(%s) order by tradingdate';
x_I = fetchmysql(sprintf(sql_str,v2_str),2);

v2_J = cellfun(@(x) ['J',num2str(x),key1],v1,'UniformOutput',false);
v2_str = ['''',strjoin(v2_J,''','''),''''];
sql_str = 'select tradingdate,close from futuredata.price_dce_data where codename in(%s) order by tradingdate';
x_J = fetchmysql(sprintf(sql_str,v2_str),2);

[inds,tref] = suscc_intersect(x_RB(:,1),x_I(:,1),x_J(:,1));
P = [x_RB(inds(:,1),2),x_I(inds(:,2),2),x_J(inds(:,3),2)];

P= cell2mat(P);
tref_str = tref;
tref = datenum(tref);
ind = tref>=datenum(2014,7,28)&tref<=datenum(2016,11,30);
tref = tref(ind);
P = P(ind,:);
tref_str = tref_str(ind);;


[~,ia] = intersect(x_RB(:,1),tref_str);
P(:,1) = cell2mat(x_RB(ia,2));

v2 = P(:,1)-P(:,2)*1.6-P(:,3)*0.5;

plot(tref,v2,'LineWidth',2)
datetick('x','yymm')
axis tight