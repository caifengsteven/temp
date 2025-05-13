%��չ�������ڻ�
clear
close all

t0 = '2011-01-04';
tt = '2018-12-30';

code_all = {'DCE','SHFE','CZCE'};

type1_sel = 1;%ѡ����Ʒ������ 1 ������ 2������ 3֣����
infos = '����';%ѡ������� �����Ϣ���Լ����ĵ�import cm data fun�еĶ���

[f_name,M_V] = import_cm_data();

type1 = code_all{type1_sel};
type2 = f_name{type1_sel}(infos);

db_name = 'futuredata';
window =20;
window2 =5;

stop_value = [2.5,1.5]/100;

multi_val = M_V{type1_sel}(infos);


cash0=1000000;
%���Ƹ� 'RB'

%��ȡ����
sql_str = ['select tradingdate,open,close from futuredata.price_if_data',10,...
    'where variety0=''%s'' and variety=''%s'' and tradingdate>=''%s''',10,...
    'and tradingdate<=''%s'' order by tradingdate'];
sql_str = sprintf(sql_str, type1,type2,t0,tt);

x = fetchmysql(sql_str,2);
%time
tref_str = x(:,1);
tref = datenum(x(:,1));
%price
P_o = cell2mat(x(:,2));
P = cell2mat(x(:,3));
%����ָ��
%�в����
P_ema = EMA(P,window);
delta = log(P)-log(P_ema);
T = length(delta);
delta_m = zeros(T,1);
for i = window+1:T
    sub_wid = i-window:i;
    delta_m(i) = (delta(i) - mean(delta(sub_wid)))/std(delta(sub_wid));
end
stop_sel = zeros(size(delta));
stop_sel(delta_m>1) =stop_value(1);
stop_sel(delta_m<=1) =stop_value(2);


T = length(P);
RVol = zeros(T,1);
for i = window:T
    sub_wid = i-window+1:i;
    RVol(i) = std(P(sub_wid)-P_ema(sub_wid))*sqrt(window);
end

RVol_ema = EMA(RVol,window2);
ind = zeros(T,1);

for i = window+1:T-1
    if RVol(i)>RVol_ema(i)
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
    %ind(i+1) = ind(i);    
end

V = bac_test_CTA_update4(ind,[P_o,P],multi_val,ones(size(ind)),100000,4/10000,stop_sel);

figure;
bpcure_plot_update(tref,V);
title(infos);

[v,v_str,sta_val] = curve_static(V);



function [f_name,M_V] = import_cm_data()
%������
f_name{1}= containers.Map({'��һ';'����';'����';'����';'�����';'����';'���׵��';'����';'��ά��';'���ϰ�';'����ϩ';'������ϩ';'�۱�ϩ';'��̿';'��ú';'����ʯ';'�Ҷ���'},...
    {'A';'B';'M';'Y';'P';'C';'CS';'JD';'FB';'BB';'L';'V';'PP';'J';'JM';'I';'EG'});
M_V{1} = containers.Map({'��һ';'����';'����';'����';'�����';'����';'���׵��';'����';'��ά��';'���ϰ�';'����ϩ';'������ϩ';'�۱�ϩ';'��̿';'��ú';'����ʯ';'�Ҷ���'},...
    [10,10,10,10,10,10,10,5,500,500,5,5,5,100,60,100,10]);
%��������Ӧ����
f_name{2}= containers.Map({'ͭ';'��';'п';'Ǧ';'��';'��';'�ƽ�';'����';'���Ƹ�';'�߲�';'�������';'ԭ��';'ȼ����';'����';'��Ȼ��';'ֽ��'},...
    {'CU';'AL';'ZN';'PB';'NI';'SN';'AU';'AG';'RB';'WR';'HC';'SC';'FU';'BU';'RU';'SP'});
M_V{2} = containers.Map({'ͭ';'��';'п';'Ǧ';'��';'��';'�ƽ�';'����';'���Ƹ�';'�߲�';'�������';'ԭ��';'ȼ����';'����';'��Ȼ��';'ֽ��'},...
    [5,5,5,5,1,1,1000,15,10,10,10,1000,50,10,10,10]);
%֣������Ӧ����
f_name{3} = containers.Map({'�޻�','����','����','����','ǿ��','Ӳ��','����','����','ǿ��','����','����','�Ͳ���','������',...
    '����ú','����','�״�','����','����','�̹�','��ɴ','ƻ��','PTA'},...
    {'CF','ER','RO','SR','WS','WT','OI','RI','WH','FG','PM','RS','RM',...
    'TC','JR','MA','LR','SF','SM','CY','AP','TA'});
M_V{3} = containers.Map({'�޻�','����','����','����','ǿ��','Ӳ��','����','����','ǿ��','����','����','�Ͳ���','������',...
    '����ú','����','�״�','����','����','�̹�','��ɴ','ƻ��','PTA'},...
    [5,20,10,10,20,20,10,20,20,20,50,10,10,100,20,10,20,5,5,5,10,5]);
end