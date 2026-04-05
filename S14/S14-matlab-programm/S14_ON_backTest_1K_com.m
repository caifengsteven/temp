%{
打分策略
时间序列动量、横截面动量和展期收益率构建三因子模型，并依
次加入基差动量和仓单变化率因子构建四因子和五因子模型，其中时间序列动量、横截
面动量因子的回看期均为 40 个交易日，展期收益率因子的期限结构类型为 TS4，基差
动量因子和仓单变化率因子的回看期分别为120和90个交易日，持有期H将作为参数。
%}

clear
key_str = '打分策略';

mod = 5;%3因子还是5因子
R = 30;
H = 10;

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

temp_re1 = cell(T,1);
temp_re2 = temp_re1;
temp_re3 = temp_re1;
temp_re4 = temp_re1;
temp_re5 = temp_re1;
temp_y_re = temp_re1;
temp_vol_re = temp_re1;

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
    
    [r,sub_tref3] = get_momentum(symbol,40);
    [~,ib,ia] = intersect(tref_num,sub_tref3,'stable');
    %r_re1(ib,symbol_sel) = r(ia);
    temp_re1{symbol_sel} = [ib,r(ia)];
    
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
end


%com
T_tref = length(tref);
m_num = 5;
m_num_2 = floor(H/m_num);
y_bac = zeros(T_tref,m_num);
ind_ini = find(sum(y_re,2),1);
if ind_ini<R
    ind_ini = (R+1);
end
for i0 = 1:m_num
    for i = ind_ini+(i0-1)*m_num_2:H:T_tref
        %1/K
        %选定数据
        ind_sel0 = find(~eq(y_re(i,:),0)&vol_re(i,:)>10000);
        
        sub_r2 = r_re2(i-1,ind_sel0);
        sub_r3 = r_re3(i-1,ind_sel0);
        sub_r4 = r_re4(i-1,ind_sel0);
        [~,ia2] = sort(sub_r2);
        [~,ia3] = sort(sub_r3);
        [~,ia4] = sort(sub_r4);
        sub_r5 = r_re5(i-1,ind_sel0);
        [~,ia5] = sort(sub_r5);
        sub_r1 = r_re1(i-1,ind_sel0);
        [~,ia1] = sort(sub_r1);

        k_score = zeros(size(y_re(1,:)));
        for j = 1:mod
            ia = eval(sprintf('ia%d',j));
            if eq(j,1)
                k_score(ia>0) = 1;
                k_score(ia<0) = -1;
            else
                if length(ia)>=5
                    num1 = floor(length(ia)*0.2);
                    ia1 = ia(1:num1);
                    ind_sel1 = ind_sel0(ia1);
                    ia2 = ia(end-num1+1:end);
                    ind_sel2 = ind_sel0(ia2);
                    k_score(ind_sel1) = k_score(ind_sel1)-1;
                    k_score(ind_sel2) = k_score(ind_sel2)+1;
                end
            end            
        end
        
        ind_sel1 = find(k_score<0);
        ind_sel2 = find(k_score>0);
        
        %归一化多、空权重
        sub_w = zeros(size(k_score));
        sub_w(k_score>0) = k_score(k_score>0)./sum(k_score(k_score>0));
        sub_w(k_score<0) = k_score(k_score<0)./sum(k_score(k_score<0));
        
        %获取收益率数据,并平均
        sub_ind = i:(i+H-1);
        sub_ind(sub_ind>T_tref) = [];
        %多
        sub_y_r_m = y_re(sub_ind,ind_sel2);    
        %手续费
        sub_y_r_m(1,:) = sub_y_r_m(1,:)-3/10000;
        sub_y_r_m(end,:) = sub_y_r_m(end,:)-3/10000;
        temp = sub_w(ind_sel2).*cumprod((1+sub_y_r_m));
        temp = [1;sum(temp,2)];
        temp_m = temp(2:end)./temp(1:end-1)-1;
        if ~isempty(ind_sel1)
            %空
            sub_y_r = y_re(sub_ind,ind_sel1);    
            %手续费
            sub_y_r([1,end],:) = sub_y_r([1,end],:);
            temp = sub_w(ind_sel1).*cumprod((1+sub_y_r));
            temp = [1;sum(temp,2)];
            temp = temp(2:end)./temp(1:end-1)-1;
        else
            temp=0;
        end
        y_bac(sub_ind,i0) = temp_m-temp;
    end
end
ind = tref_num>datenum(2010,1,1);
y_bac1 = y_bac(ind,:);
tref_num1 = tref_num(ind);

y_bac_t = 1/m_num*cumprod(y_bac1+1);
y_bac_t = sum(y_bac_t,2);
bpcure_plot_updateV2(tref_num1,y_bac_t);
[v,v_str,sta_val] = curve_static(y_bac_t)
%时间序列动量因子
function [r,tref] = get_momentum(symbol,N)
sql_str = ['select tradingdate,close_price from futuredata.YQ_future_rehabilitation_data ',...
    'where symbol = ''%s'' and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,strjoin(symbol,'.')),2);
y = cell2mat(x(:,2));
r = zeros(size(y));
r(N+1:end) = y(N+1:end)./y(1:end-N)-1;
tref = datenum(x(:,1));
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


