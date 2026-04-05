%{
复合信号策略策略
%}
clear
key_str = 'S14策略';

%t0 = '2005-01-01';
%tref = yq_methods.get_tradingdate(t0,datestr(now,'yyyy-mm-dd'));
tref = fetchmysql('select distinct(tradedate) from yuqerdata.yq_MktMFutdGet where tradeDate>=''2005-01-01''',2);
tref_num = datenum(tref);

%list
%获取品种
sql_str = ['select exchangeCD,contractObject,secShortName,contMultNum from ',...
    ' yuqerdata.yq_FutuGet where exchangeCD in (''XDCE'',''XSGE'',''XZCE'')'];
x = fetchmysql(sql_str,2);
y = cellfun(@(x,y) [x,'.',y],x(:,1),x(:,2),'UniformOutput',false);
[y,ia] = unique(y);
x = x(ia,:);

sql_str2 = ['select exchangeCD,contractObject from yuqerdata.yq_MktMFutdGet ',...
    ' where exchangeCD in (''XDCE'',''XSGE'',''XZCE'') and tradeDate = ''%s''',...
    ' and mainCon=1'];
x1 = fetchmysql(sprintf(sql_str2,tref{end}),2);
y1 = cellfun(@(x,y) [x,'.',y],x1(:,1),x1(:,2),'UniformOutput',false);
[symbol0,ia] = intersect(y,y1);
x = x(ia,:);
sy_info0 = x(:,3);
T = length(sy_info0);
for i = 1:T
    sub_ind = isletter(sy_info0{i});
    sy_info0{i} = sy_info0{i}(sub_ind);
end
M = cell2mat(x(:,4));
T = length(symbol0);

%这段需要修改为并行
y_re = zeros(length(tref),T);
vol_re = y_re;
r_re1 = y_re;
r_re2 = y_re;
r_re3 = y_re;
r_re4 = y_re;
r_re5 = y_re;
close_price = y_re;
close_price_r = y_re;

temp_re1 = cell(T,1);
temp_re2 = temp_re1;
temp_re3 = temp_re1;
temp_re4 = temp_re1;
temp_re5 = temp_re1;
temp_y_re = temp_re1;
temp_vol_re = temp_re1;
temp_close_price = temp_re1;
temp_close_price_r = temp_re1;

parfor symbol_sel = 1:T
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');
    [cash_flow,sub_tref]=get_bac_data_yuqer_update(symbol,M(symbol_sel),0.2);

    [~,ia] = intersect(tref_num,sub_tref,'stable');
    %y_re(ia,symbol_sel) = [0;cash_flow(2:end)./cash_flow(1:end-1)-1];
    temp_y_re{symbol_sel} = [ia,[0;cash_flow(2:end)./cash_flow(1:end-1)-1]];
    
    [v,sub_tref2] = get_vol_data(symbol);
    [~,ib] = intersect(tref_num,sub_tref2,'stable');
    %vol_re(ib,symbol_sel) = movmean(v,[20,0]);
    temp_vol_re{symbol_sel} = [ib,movmean(v,[20,0])];
    
    [r,sub_tref3,sub_close_price] = get_momentum(symbol,40);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    %r_re1(ib,symbol_sel) = r(ia);
    %close_price(ib,symbol_sel) = sub_close_price(ia,1);
    %close_price_r(ib,symbol_sel) = sub_close_price(ia,2);
    temp_re1{symbol_sel} = [ib,r(ia)];
    temp_close_price{symbol_sel} = [ib,sub_close_price(ia,1)];
    temp_close_price_r{symbol_sel} = [ib,sub_close_price(ia,2)];
    [r,sub_tref3] = get_sectional_momentum(symbol,40);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    %r_re2(ib,symbol_sel) = r(ia);
    temp_re2{symbol_sel} = [ib,r(ia)];
    
    [r,sub_tref3] = get_roll_return_yq(symbol,4);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    %r_re3(ib,symbol_sel) = r(ia);
    temp_re3{symbol_sel} = [ib,r(ia)];
    
    [r,sub_tref3] = get_basismomentum_return(symbol,120);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    %r_re4(ib,symbol_sel) = r(ia);
    temp_re4{symbol_sel} = [ib,r(ia)];
    
    [r,sub_tref3] = get_warehouse(symbol,90);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    %r_re5(ib,symbol_sel) = -r(ia);
    temp_re5{symbol_sel} = [ib,-r(ia)];
    
    sprintf('%s 载入数据: %d-%d',key_str,symbol_sel,T)
end
for symbol_sel = 1:T
    sub_re1 = temp_re1{symbol_sel};
    r_re1(sub_re1(:,1),symbol_sel) = sub_re1(:,2);
    
    sub_re = temp_re2{symbol_sel};
    r_re2(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_re3{symbol_sel};
    r_re3(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_re4{symbol_sel};
    r_re4(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_re5{symbol_sel};
    r_re5(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_y_re{symbol_sel};
    y_re(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_vol_re{symbol_sel};
    vol_re(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_close_price{symbol_sel};
    close_price(sub_re(:,1),symbol_sel) = sub_re(:,2);
    
    sub_re = temp_close_price_r{symbol_sel};
    close_price_r(sub_re(:,1),symbol_sel) = sub_re(:,2);
end
%权重和打分
%复合信号策略策略
sub_w = cell(4,1);
k_score = cell(4,1);
[sub_w{1},k_score{1}] = get_volitylity_signal(tref,y_re,r_re1,r_re2,r_re3,r_re4,r_re5,vol_re,close_price,close_price_r);
%多因子打分
[sub_w{2},k_score{2}]  = get_score_signal(tref,y_re,r_re2,r_re3,r_re4,r_re5,vol_re);
%3因子1/K加权
[sub_w{3},k_score{3}]  = get_idiosyncractic_signal(tref,y_re,r_re1,r_re2,r_re3,r_re4,r_re5,vol_re);
%5因子1/K加权法
[sub_w{4},k_score{4}]  = get_1K_signal(tref,y_re,r_re1,r_re2,r_re3,r_re4,r_re5,vol_re);
methods = {'复合信号策略策略','多因子打分','3因子1/K加权','5因子1/K加权法'};
f_str1 = containers.Map([-1,0,1],{'做空','平仓','做多'});
info1 = cell(length(sub_w{1}),4);
x = info1;
for i = 1:4
    for j = 1:length(sub_w{1})
        sub_x = k_score{i}(j);
        sub_x(sub_x>0) = 1;
        sub_x(sub_x<0) = -1;
        info1{j,i} = f_str1(sub_x);
    end
    x(:,i) = num2cell(sub_w{i}*100);
end
X = cell(length(sub_w{1}),8);
X(:,1:2:end) = info1;
X(:,2:2:end) = x;
t_str = sprintf('S14策略操作%s',tref{end});
t_c = cell(size(X(1,:)));
t_c(1:2:end) = methods;
t_c(2:2:end) = methods;

t_r =cellfun(@(x,y) [x,'-',y],sy_info0,symbol0,'UniformOutput',false);
gui_result(X,t_str,t_c,t_r)

re = [[{' '};t_r],[t_c;X]];
re = cell2table(re);
fn = sprintf('%s.csv',t_str);
writetable(re,fn);

%时间序列动量因子
function [r,tref,add_v] = get_momentum(symbol,N)
sql_str = ['select tradingdate,close_price from futuredata.YQ_future_rehabilitation_data ',...
    'where symbol = ''%s'' and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));
add_v = [y,[0;y(2:end)./y(1:end-1)]];
end
%截面动量因子
function [r,tref] = get_sectional_momentum(symbol,N)
sql_str = ['select tradingdate,close_price from futuredata.YQ_future_rehabilitation_data ',...
    'where symbol = ''%s'' and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));
end
%展期收益率因子
function [r,tref] = get_roll_return_yq(symbol,N)
sql_str = ['select tradingdate,R1,R2,R3,R4 from futuredata.yuqer_future_rollreturn ',...
    'where  symbol = ''%s'' ',...
    'and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,symbol{2}),2);
tref = datenum(x(:,1));
r = cell2mat(x(:,N+1));

end
%基差动量
function [r,tref] = get_basismomentum_return(symbol,N,mod)
if nargin < 3
    mod = 1;
end
sql_str = ['select tradingdate,R1,R2,R3,R4,R5 from futuredata.yuqer_future_basis_momentum ',...
    'where symbol = ''%s'' and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,symbol{2}),2);
y = cell2mat(x(:,2:end));
y(y>0.1) = 0.1;
y(y<-0.1) = -0.1;
y = cumprod(1+y);
r = zeros(size(y));
r(N+1:end,:) = y(N+1:end,:)./y(1:end-N,:);%累积收益率
%当月 次月 主力 次主力 最远月
tref = datenum(x(:,1));
if eq(mod,1)
    r = r(:,1)-r(:,3);
elseif eq(mod,2)
    r = r(:,3)-r(:,4);
end

end
%仓单因子
function [r,tref] = get_warehouse(symbol,R)
    %R = 240;
    sql_str = ['select tradedate,wrvol from futuredata.yq_warehousefactor_data ',...
        'where contractobject = ''%s'' and tradedate>=''2005-01-01'' order by tradedate'];
    x = fetchmysql(sprintf(sql_str,symbol{2}),2);
    if ~isempty(x)
        y = cell2mat(x(:,2:end));
        r = zeros(size(y));
        for i = R+1:length(y)
            r(i) = (y(i)-y(i-R))/y(i-R);
        end
        r(isnan(r)|isinf(r)) = 0;
        tref = datenum(x(:,1));
    else
        r = [];
        tref = [];
    end
end


function [x,tref] = get_vol_data(symbol)
sql_str = ['select tradedate,turnoverVol from yuqerdata.yq_MktMFutdGet ',...
        'where exchangeCD=''%s'' and contractObject=''%s''and openprice is not null and closeprice is not null ',...
        'and tradedate>=''2005-01-01'' and mainCon=1 order by tradedate'];
y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
x = cell2mat(y_jj(:,2));
tref = datenum(y_jj(:,1));
end


