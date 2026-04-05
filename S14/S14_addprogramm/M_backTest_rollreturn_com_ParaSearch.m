%{
做多展期收益率排名前 20%的品种，作为多头组合；做
多展期收益率排名后 20%的品种，作为空头组合；做多展期收益率排名前 20%的品种，
做空展期收益率排名后 20%的品种，作为多空组合。

%}

clear
%close all

R_pool = 1:4;
H_pool = 5:5:20;
T1 = length(R_pool);
T2 = length(H_pool);

paras_all = zeros(T1*T2,4);
k = 1;
for i = 1:T1
    para1 = R_pool(i);
    for j = 1:T2
        para2 = H_pool(j);
        paras_all(k,:) = [i,j,para1,para2];
        k = k + 1;
    end
end
T = size(paras_all,1);

re0 = cell(T,1);

parfor i = 1:T
    sub_para = paras_all(i,:);
    v1 = sub_para(1);
    v2 = sub_para(2);
    R = sub_para(3);
    H=sub_para(4);
    y_bac_t =  get_commentum_bac(R,H);
    [v,v_str,sta_val] = curve_static0(y_bac_t);
    re0{i} = [sta_val.nh,sta_val.sharp,sta_val.drawdown];
end

nianhua_re = zeros(T1,T2);
sharp_re = zeros(T1,T2);
maxdrawdown_re = zeros(T1,T2);
for i = 1:T
    sub_para = paras_all(i,:);
    v1 = sub_para(1);
    v2 = sub_para(2);
    sub_x = re0{i};
    nianhua_re(v1,v2) =sub_x(1);
    sharp_re(v1,v2) =sub_x(2);
    maxdrawdown_re(v1,v2) =sub_x(3);
end

sub_x_var = cellstr(num2str(R_pool'));
sub_y_var = cellstr(num2str(H_pool'));
subplot(1,3,1)
h = heatmap(nianhua_re*100);
h.XData = sub_y_var;
h.YData = sub_x_var;
title('年华收益%')
subplot(1,3,2)
h = heatmap(sharp_re);
h.XData = sub_y_var;
h.YData = sub_x_var;
title('Sharp')
subplot(1,3,3)
h = heatmap(maxdrawdown_re);
h.XData = sub_y_var;
h.YData = sub_x_var;
title('最大回撤')
function y_bac_t =  get_commentum_bac(R,H);
sql_str = ['select distinct tradedate from futuredata.yuqer_fusharedata where ',...
    'tradedate <= ''2017-03-01'' and tradedate>=''2005-01-01''',...
    ' order by tradedate'];
tref = fetchmysql(sql_str,2);
tref_num = datenum(tref);
%主要差异
%{
主力合约数据更换
交易所代码更换


%}
%list
fix_fushare_info = containers.Map({'CZCE','SHFE','DCE'},{'XZCE','XSGE','XDCE'});


[~,~,x] = xlsread('future_info.xlsx','sheet3');
symbol0 = cellfun(@(x,y) [x,'.',y],x(:,1),x(:,2),'UniformOutput',false);
sy_info0 = x(:,3);
M = cell2mat(x(:,6));
T = length(symbol0);
y_re = zeros(length(tref),T);
vol_re = y_re;
r_re = y_re;
for symbol_sel = 1:T
    symbol = symbol0{symbol_sel};
    sy_info = sy_info0{symbol_sel};

    symbol = strsplit(symbol,'.');
    symbol{1} = fix_fushare_info(symbol{1});
    [cash_flow,sub_tref]=get_bac_data_yuqer(symbol,M(symbol_sel),0.25);

    [~,ia] = intersect(tref_num,sub_tref,'stable');
    y_re(ia,symbol_sel) = [0;cash_flow(2:end)./cash_flow(1:end-1)-1];
    
    [v,sub_tref2] = get_vol_data_yuqer(symbol);
    [~,ib] = intersect(tref_num,sub_tref2,'stable');
    vol_re(ib,symbol_sel) = movmean(v,[20,0]);
    
    %用展期收益率代替
    [r,sub_tref3] = get_roll_return_yq(symbol,R);
    [~,ib] = intersect(tref_num,sub_tref3,'stable');
    r_re(ib,symbol_sel) = r;
    
    
    sprintf('BacTest %d-%d',symbol_sel,T)
end



%com
T_tref = length(tref);
y_bac = zeros(T_tref,1);

ind_ini = find(sum(y_re,2),1);
if ind_ini<2
    ind_ini = 2;
end
for i = ind_ini:H:T_tref
    %选定数据
    ind_sel0 = find(~eq(r_re(i-1,:),0)&vol_re(i,:)>10000&r_re(i,:)<10000&r_re(i-1,:)<10000);
    sub_r = r_re(i-1,ind_sel0);
    if length(sub_r)>5
        [~,ia] = sort(sub_r);
        ia1 = ia(1:floor(end*0.2));
        ind_sel1 = ind_sel0(ia1);
        ia2 = ia(end-floor(end*0.2)+1:end);
        ind_sel2 = ind_sel0(ia2);
    else
        ind_sel1 = [];
        ind_sel2 = ind_sel0;
        
    end    
    
    %获取收益率数据,并平均
    sub_ind = i:(i+H-1);
    sub_ind(sub_ind>T_tref) = [];
    
    %多
    sub_y_r_m = y_re(sub_ind,ind_sel2);    
    %手续费
    sub_y_r_m(1,:) = sub_y_r_m(1,:)-3/10000;
    sub_y_r_m(end,:) = sub_y_r_m(end,:)-3/10000;
    temp = 1/size(sub_y_r_m,2)*cumprod((1+sub_y_r_m));
    temp = [1;sum(temp,2)];
    temp_m = temp(2:end)./temp(1:end-1)-1;
    if ~isempty(ind_sel1)
        %空
        sub_y_r = y_re(sub_ind,ind_sel1);    
        %手续费
        sub_y_r([1,end],:) = sub_y_r([1,end],:);
        temp = 1/size(sub_y_r,2)*cumprod((1+sub_y_r));
        temp = [1;sum(temp,2)];
        temp = temp(2:end)./temp(1:end-1)-1;
    else
        temp=0;
    end
    y_bac(sub_ind) = temp_m-temp;
    
    
    
end


y_bac_t = cumprod(y_bac+1);
bpcure_plot_updateV2(tref_num,y_bac_t);
end

function [r,tref] = get_roll_return_yq(symbol,N)
sql_str = ['select tradingdate,R1,R2,R3,R4 from futuredata.yuqer_future_rollreturn ',...
    'where exchangeCD = ''%s'' and symbol = ''%s'' and tradingdate <= ''2017-03-01''  ',...
    'and tradingdate>=''2005-01-01'' order by tradingdate'];
x = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
tref = datenum(x(:,1));
r = cell2mat(x(:,N+1));

end


function [x,tref] = get_vol_data_yuqer(symbol)
sql_str = ['select tradedate,turnoverVol from futuredata.yuqer_fusharedata ',...
        'where exchangeCD=''%s'' and contractObject=''%s'' and openprice is not null ',...
        'and closeprice is not null and tradedate <= ''2017-03-01''  ',...
        'and tradedate>=''2005-01-01'' and mainCon=1 order by tradedate'];
y_yuqer = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);
x = cell2mat(y_yuqer(:,2));
tref = datenum(y_yuqer(:,1));
end



