%{
FICC 系列研究之二 —— 主力合约合成
2.1 回测品种选择
%}

clear

symbol = {'SHFE.FU','SHFE.BU','SHFE.RB'};
sy_info = {'燃油','沥青','螺纹钢'};
symbol = symbol{3};

symbol = strsplit(symbol,'.');

tref = fetchmysql(sprintf('select distinct(tradingdate) from futuredata.price_%s_data where variety=''%s''',symbol{1},symbol{2}),2);
sql_str = 'select codename,close,volume from futuredata.price_%s_data where variety=''%s'' and tradingdate = ''%s'' order by volume desc';
codename1 = cell(size(tref)); %主力合约code
codename2 = codename1; %持仓量最大合约code
y = zeros(size(tref));
T = length(y);
for i = 1:T
    sub_sql_str = sprintf(sql_str,symbol{1},symbol{2},tref{i});
    sub_x = fetchmysql(sub_sql_str,2);
    sub_x_data = cell2mat(sub_x(:,2:end));
    codename2(i) = sub_x(1,1);
    if i<=2
        y(i) = sub_x_data(1,1);
        codename1(i) = sub_x(1,1);
    else
        %判断是否切换
        %连续两天主力合约不是最大持仓量合约，并且持仓量合约不能后退
        if all(~strcmp(codename2(i-1:i),codename1{i-1})) && strcmp(codename2(i),codename2(i-1)) && str2double(codename2{i}(length(symbol)+1:end)) >str2double(codename1{i-1}(length(symbol)+1:end))
            codename1(i) = codename2(i);
            y(i) = sub_x_data(1,1);
        else            
            ia = strcmp(sub_x(:,1),codename1(i-1));
            if any(ia)
                y(i) = sub_x_data(ia,1);
                codename1(i) = codename1(i-1);
            else
                y(i) = sub_x_data(1,1);
                codename1(i) = codename2(1);
            end
            
        end
        
    end    
    sprintf('%d-%d',i,T)   
end

%获取掘金主连数据
sql_str = 'select tradingdate,close from futuredata.price_if_data where variety0=''%s'' and variety=''%s'' and tradingdate <= ''2017-03-01'' order by tradingdate';
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);


[~,ia,ib] = intersect(tref,y_jj(:,1));

tref0 = tref(ia);

Y = [y(ia),cell2mat(y_jj(:,2))];
yyaxis left
plot(Y);
yyaxis right
bar(Y(:,1)-Y(:,2));

set(gca,'XTickLabelRotation',90);
set(gca,'XTick',floor(linspace(1,length(tref0),40)),'xlim',[1,length(tref0)]);
set(gca,'XTickLabel',tref0(floor(linspace(1,length(tref0),40))));
%datetick('x','yyyymmdd','keeplimits');
set(gca,'fontsize',12);



% x = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
% 
% tref = datenum(x(:,1));
% y = cell2mat(x(:,3));
% x = cell2mat(x(:,2));
% 
% 
% yyaxis left
% bar(y);
% yyaxis right
% plot(x,'LineWidth',2);
% set(gca,'XTickLabelRotation',90);
% set(gca,'XTick',floor(linspace(1,length(tref),40)),'xlim',[1,length(tref)]);
% set(gca,'XTickLabel',cellstr(datestr(tref(floor(linspace(1,length(tref),40))),'yyyymmdd')));
% %datetick('x','yyyymmdd','keeplimits');
% set(gca,'fontsize',12);