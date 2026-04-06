%{
Á¶½¹²úÒµ

f_name_r{1}= containers.Map({'¶¹Ò»';'¶¹¶þ';'¶¹ÆÉ';'¶¹ÓÍ';'×ØéµÓÍ';'ÓñÃ×';'ÓñÃ×µí·Û';'¼¦µ°';'ÏËÎ¬°å';'½ººÏ°å';'¾ÛÒÒÏ©';'¾ÛÂÈÒÒÏ©';'¾Û±ûÏ©';'½¹Ì¿';'½¹Ãº';'Ìú¿óÊ¯';'ÒÒ¶þ´¼'},...
    {'A';'B';'M';'Y';'P';'C';'CS';'JD';'FB';'BB';'L';'V';'PP';'J';'JM';'I';'EG'});

%ÉÏÉÌËù¶ÔÓ¦Ãû³Æ
f_name_r{2}= containers.Map({'Í­';'ÂÁ';'Ð¿';'Ç¦';'Äø';'Îý';'»Æ½ð';'°×Òø';'ÂÝÎÆ¸Ö';'Ïß²Ä';'ÈÈÔþ¾í°å';'Ô­ÓÍ';'È¼ÁÏÓÍ';'Á¤Çà';'ÌìÈ»Ïð½º';'Ö½½¬'},...
    {'CU';'AL';'ZN';'PB';'NI';'SN';'AU';'AG';'RB';'WR';'HC';'SC';'FU';'BU';'RU';'SP'});

%Ö£ÉÌËù¶ÔÓ¦Ãû³Æ
f_name_r{3} = containers.Map({'ÃÞ»¨','ÔçôÌ','²ËÓÍ','°×ÌÇ','PTA','Ç¿Âó','Ó²Âó','¼×´¼','²ËÓÍ','ÔçôÌ','Ç¿Âó','²£Á§','ÆÕÂó','ÓÍ²Ë×Ñ','²Ë×ÑÆÉ',...
    '¶¯Á¦Ãº','¶¯Á¦Ãº','¾¬µ¾','¼×´¼','ÍíôÌ','¹èÌú','ÃÌ¹è','ÃÞÉ´','Æ»¹û','PTA'},...
    {'CF','ER','RO','SR','PTA','WS','WT','ME','OI','RI','WH','FG','PM','RS','RM',...
    'TC','ZC','JR','MA','LR','SF','SM','CY','AP','TA'});
»ñÈ¡Á¬1Á¬3Á¬9Êý¾Ý
num=1+Á¬Êý
ÈýÖÖ×éºÏÐÅºÅ
Á¶½¹ÀûÈó

%}
clear

codes = {'J','JM'};
tb_names = {'dce','dce'};
scale_num = [60,100];
num = [5,9]+1;

t1 = datenum(2013,10,11);
t2 = datenum(2016,12,30);
%»ñÈ¡Ö÷Á¬Êý¾Ý
T_code =length(codes); 
X = cell(T_code,1);
sql_str = 'select tradingdate,open,close from futuredata.price_if_data where variety=''%s''';

for i = 1:T_code
    sub_sql_str = sprintf(sql_str,codes{i});
    X{i} = fetchmysql(sub_sql_str,2);
end

[inds,tref] = suscc_intersect(X{1}(:,1),X{2}(:,1));

for i = 1:length(codes)
    X{i}=cell2mat(X{i}(inds(:,i),2:3));
end

tref = datenum(tref);
ind = tref>=t1&tref<=t2;
tref = tref(ind);
for i = 1:length(X)
    X{i}=X{i}(ind,:);
end
T = length(tref);

sql_str2 = 'select close from futuredata.price_%s_data where tradingdate=''%s'' and variety=''%s'' order by codename';

%»ñÈ¡Á¬ÐøÊý¾Ý
Y = cell(T,T_code);
parfor i = 1:T
    sub_vec = datevec(tref(i));
    sub_t = datestr(tref(i),'yyyy-mm-dd');
    for j = 1:T_code
        sub_str = sprintf(sql_str2,tb_names{j},sub_t,codes{j});
        temp = fetchmysql(sub_str);
        Y(i,j) = {temp(num)};
    end
    i
end
%¼ÆËãÐÅºÅ
for num_sel = 1:length(num)
    P = cellfun(@(x) x(num_sel),  Y);

    sub_x = P(:,1)-P(:,2)*1.4;
    %d1 = movavg(d,'simple',10);
    sub_mx = zeros(size(sub_x));
    for i = 11:length(sub_mx)
        sub_mx(i) = mean(sub_x(i-10:i-1));
    end


    %sd1 = movstd(d,[10,0]);
    sub_stdx = zeros(size(sub_x));
    for i = 11:length(sub_stdx)
        sub_stdx(i) = std(sub_x(i-10:i-1));
    end
    %get signal
    T = length(sub_x);
    sub_signal = zeros(T,1);
    if eq(num_sel,1)
        signal = zeros(T,length(num));
    end
    for i = 10:T
        %temp = d(i)>=d1(i)+sd1(i)&d(i)<=d1(i)+sd1(i)*1.2&d(i)<d(i-1)&d(i-1)<d(i-2);
        temp = sub_x(i)>=sub_mx(i)+sub_stdx(i)&sub_x(i)<=sub_mx(i)+sub_stdx(i)*1.2&sub_x(i)<=sub_x(i-1);
        if temp
            sub_signal(i) = 1;
            continue
        end

        if sub_x(i)<=sub_mx(i)
            sub_signal(i)=0;
            continue
        end

        sub_signal(i) = sub_signal(i-1);    
    end
    signal(:,num_sel) = sub_signal;
end

%×ÛºÏÐÅºÅ
signal = sum(signal,2);
signal(signal>1) = 1;
%¿ªÊ¼»Ø²â
cash0 = 1000000;

VB = zeros(T,1);
VU = VB;
re_curve = VB;

lots_num = 3;
stop_value = -5/100;
re_curve(1) = cash0;
i = 1;
while i <T
    i = i + 1;
    if eq(signal(i-1),0)
        VB(i) = 0;
        VU(i) = 0;
    elseif eq(signal(i-1),-1)
        VB(i) = VB(i) +(X{1}(i,2)-X{1}(i,1))*scale_num(1)*lots_num;
        %VB(i) = VB(i) + (x(i,2)-x(i,1))*v1*lots_num;
        for j = 2:length(X)
            VU(i) = VU(i) - (X{j}(i,2)-X{j}(i,1))*scale_num(j)*lots_num;
        end
    else
        VB(i) = VB(i) -(X{1}(i,2)-X{1}(i,1))*scale_num(1)*lots_num;
        %VB(i) = VB(i) + (x(i,2)-x(i,1))*v1*lots_num;
        for j = 2:length(X)
            VU(i) = VU(i) + (X{j}(i,2)-X{j}(i,1))*scale_num(j)*lots_num;
        end
    end
    
    re_curve(i) = re_curve(i-1)+VB(i)+VU(i);
    if re_curve(i)/re_curve(i-1)-1<stop_value
        %vv(i:i+10) = vv(i-1)*(1+stop_v);
        re_curve(i:i+10) = re_curve(i);
        i = i + 10;        
    end

end



bpcure_plot_update(tref,re_curve);

