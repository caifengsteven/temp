%{
PP 甲醇
f_name_r{1}= containers.Map({'豆一';'豆二';'豆粕';'豆油';'棕榈油';'玉米';'玉米淀粉';'鸡蛋';'纤维板';'胶合板';'聚乙烯';'聚氯乙烯';'聚丙烯';'焦炭';'焦煤';'铁矿石';'乙二醇'},...
    {'A';'B';'M';'Y';'P';'C';'CS';'JD';'FB';'BB';'L';'V';'PP';'J';'JM';'I';'EG'});

%上商所对应名称
f_name_r{2}= containers.Map({'铜';'铝';'锌';'铅';'镍';'锡';'黄金';'白银';'螺纹钢';'线材';'热轧卷板';'原油';'燃料油';'沥青';'天然橡胶';'纸浆'},...
    {'CU';'AL';'ZN';'PB';'NI';'SN';'AU';'AG';'RB';'WR';'HC';'SC';'FU';'BU';'RU';'SP'});

%郑商所对应名称
f_name_r{3} = containers.Map({'棉花','早籼','菜油','白糖','PTA','强麦','硬麦','甲醇','菜油','早籼','强麦','玻璃','普麦','油菜籽','菜籽粕',...
    '动力煤','动力煤','粳稻','甲醇','晚籼','硅铁','锰硅','棉纱','苹果','PTA'},...
    {'CF','ER','RO','SR','PTA','WS','WT','ME','OI','RI','WH','FG','PM','RS','RM',...
    'TC','ZC','JR','MA','LR','SF','SM','CY','AP','TA'});
获取连1连3连9数据
num=1+连数
三种组合信号
PP 甲醇
郑商所数据格式不同，不能直接用该方法
%}
clear

codes = {'PP','MA'};
tb_names = {'dce','czce'};
scale_num = [10,5];
num = [1,5,9]+1;

window_num = 5;
t1 = datenum(2014,10,12);
t2 = datenum(2016,12,30);
%获取主连数据
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
sql_str3 = 'select close from futuredata.price_%s_data where tradingdate=''%s'' and codename=''%s99''';
%获取连续数据
Y = cell(T,T_code);
parfor i = 1:T
    sub_vec = datevec(tref(i));
    sub_t = datestr(tref(i),'yyyy-mm-dd');
    for j = 1:T_code
        if eq(j,1)
            sub_str = sprintf(sql_str2,tb_names{j},sub_t,codes{j});
            temp = fetchmysql(sub_str);
            Y(i,j) = {temp(num)};
        else
            sub_str = sprintf(sql_str3,tb_names{j},sub_t,codes{j});
            temp = fetchmysql(sub_str);
            Y(i,j) = {repmat(temp,length(num),1)};
        end
    end
    i
end
%计算信号
for num_sel = 1:length(num)
    P = cellfun(@(x) x(num_sel),  Y);

    sub_x = P(:,1)-P(:,2)*3-800;
    %d1 = movavg(d,'simple',10);
    sub_mx = zeros(size(sub_x));
    for i = window_num+1:length(sub_mx)
        sub_mx(i) = mean(sub_x(i-window_num:i-1));
    end


    %sd1 = movstd(d,[10,0]);
    sub_stdx = zeros(size(sub_x));
    for i = window_num+1:length(sub_stdx)
        sub_stdx(i) = std(sub_x(i-window_num:i-1));
    end
    %get signal
    T = length(sub_x);
    sub_signal = zeros(T,1);
    if eq(num_sel,1)
        signal = zeros(T,length(num));
    end
    for i = window_num+1:T
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

%综合信号
signal = sum(signal,2);
signal(signal>1) = 1;
%开始回测
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

