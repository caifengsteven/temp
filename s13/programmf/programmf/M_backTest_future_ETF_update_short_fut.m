%回测思路是，（1）第一天开盘做多一半ETF，同时做空0.5股指期货；尾盘不做任何事
%（2）第二天早上做多0.5etf，同时做空0.5股指；下午收盘平昨日仓位；
%（3）第三日同第二日，直到交易终止。

clear

code_sel = 2;

index_name_pool = {'嘉实沪深300ETF159919.csv';'华夏上证50ETF510050.csv';'南方中证500ETF510500.csv'};
sub_etf_name = index_name_pool{code_sel}(1:end-4);
[tref_etf,open_price_etf,close_price_etf] = get_etf_data(sub_etf_name);

index_name_pool = {'沪深300股指期货','上证50股指期货','中证500股指期货'};
[tref,index_contracts_num,open_price,close_price] = get_future_data(code_sel);


[tref,ia,ib] = intersect(tref_etf,tref);
open_price_etf = open_price_etf(ia,:);
close_price_etf = close_price_etf(ia,:);

index_contracts_num = index_contracts_num(ib);
index_contracts_num = cat(1,index_contracts_num,true);
open_price = open_price(ib,:);
close_price = close_price(ib,:);

%回测
fee = 0;
[Y1,Y1_etf,Y1_g] = bac_method_com(tref,index_contracts_num,open_price,close_price,open_price_etf,close_price_etf,fee);
[Y2,Y2_etf,Y2_g] = bac_method_com(tref,index_contracts_num,open_price,close_price,open_price_etf,close_price_etf,1/10000);

obj=plot(tref,[sum(Y1,2),sum(Y2,2)],'linewidth',2);
set(gca,'XTickLabelRotation',90);
set(gca,'XTick',tref(floor(linspace(1,length(tref),20))),'xlim',tref([1,end]));
datetick('x','yyyymmdd','keepticks');
set(gca,'fontsize',12);

leg_str = {'无手续费','手续费万1'};
legend(obj,leg_str,'Location','northwest',...
    'NumColumns',length(obj),'location','best')
figure;
bpcure_plot_update(tref,sum(Y1,2))

function [Y,Y1,Y2] = bac_method_com(tref,index_contracts_num,open_price,close_price,open_price_etf,close_price_etf,fee)
    Y1 = [1,0]; %ETF %资金流，仓位
    Y2 = [1,0]; %股指期货 %资金流，仓位
    T = length(tref);
    %fee = 0;
    for i = 1:T
        if eq(i,1)
            Y1(1,:) = [0.5,0.5/(1+fee)];%做多0.5etf
            Y2(1,:) = [0.5,0.5/(1+fee)];%做空0.5股指
        else
            %股指期货主力合约更换日需要处理
            %若明日股指期货换仓日的话，尾盘不买入股指期货，空一天
            %当日换仓日的话，早盘之买入一半股指仓位
            if ~index_contracts_num(i+1)                
                %开盘
                %买多etf，买空股指
                %如果是换仓日，则不必卖出股指
                if ~index_contracts_num(i)
                    position_open_g =Y2(i-1,1)/(1+fee);
                else
                    position_open_g =Y2(i-1,1)*0.5/(1+fee);
                end
                %Y2_temp(1) = 0;
                %Y2_temp(2) = Y2_temp(2)+position_open_g;
                position_open_etf = Y1(i-1,1)/(1+fee);
                %Y1_temp(1) = 0;
                %Y1_temp(2) = Y1_temp(2)+position_open_etf;
                
                %收盘
                %卖出昨日ETF仓位
                position_close_etf = Y1(i-1,2)*close_price_etf(i)./open_price_etf(i-1)*(1-fee);
                Y1(i,:) = [position_close_etf,position_open_etf];
                %平昨日股指仓位
                position_close_g = Y2(i-1,2)*(open_price(i-1)/close_price(i))*(1-fee);
                if ~index_contracts_num(i)
                    Y2(i,:) = [position_close_g,position_open_g];
                else
                    Y2(i,:) = [Y2(i-1,1)*0.5,position_open_g];
                end

            else
                %开盘
                %买多etf，买空股指
                %如果第二天是换仓日，不做股指仓位
                position_open_g =0;
                %Y2_temp(1) = 0;
                %Y2_temp(2) = Y2_temp(2)+position_open_g;
                position_open_etf = Y1(i-1,1)/(1+fee);
                %Y1_temp(1) = 0;
                %Y1_temp(2) = Y1_temp(2)+position_open_etf;
                
                %收盘
                %卖出昨日ETF仓位
                position_close_etf = Y1(i-1,2)*close_price_etf(i)./open_price_etf(i-1)*(1-fee);
                Y1(i,:) = [position_close_etf,position_open_etf];
                %平昨日股指仓位
                position_close_g = Y2(i-1,2)*(open_price(i-1)/close_price(i))*(1-fee);
                Y2(i,:) = [position_close_g+Y2(i-1,1),position_open_g];           
            end
        end

    end
    Y = sum([Y1,Y2],2);
end

function [tref,index_contracts_num,open_price,close_price] = get_future_data(index_sel)
    %股指数据
    %index_name_pool = {'沪深300股指期货','上证50股指期货','中证500股指期货'};
    index_code = {'IF','IH','IC'};    
    index_pool = {'CFFEX','CFFEX','CFFEX'};    

    t0 = cell(size(index_code));
    t0{1} = '2014-05-01';
    tt = datenum(2019,3,1);

    key_contracts = [index_pool{index_sel},'.',index_code{index_sel}];
    if isempty(t0{index_sel})
        sql_str = 'select tradingdate,open,close from futuredata.price_if_data where  variety0 = ''%s'' and variety = ''%s'' order by tradingdate';
        sub_sql_str = sprintf(sql_str,index_pool{index_sel},index_code{index_sel});
    else
        sql_str = 'select tradingdate,open,close from futuredata.price_if_data where  variety0 = ''%s'' and variety = ''%s'' and tradingdate >= ''%s'' order by tradingdate';
        sub_sql_str = sprintf(sql_str,index_pool{index_sel},index_code{index_sel},t0{index_sel});
    end
    index_data = fetchmysql(sub_sql_str,2);
    if ~isempty(tt)
        index_data = index_data(datenum(index_data(:,1))<=tt,:);
    end

    sql_str2 = 'select tradingdate,symbol from futuredata.future_contracts_data where variety=''%s'' order by tradingdate';
    sub_sql_str2 = sprintf(sql_str2,key_contracts);
    index_contracts = fetchmysql(sub_sql_str2,2);
    [~,ia,ib] = intersect(index_data(:,1),index_contracts(:,1));
    index_data = index_data(ia,:);
    index_contracts = index_contracts(ib,:);
    index_contracts_num = cellfun(@(x) str2double(x(length(key_contracts)+1:end)),index_contracts(:,end));
    index_contracts_num = [0;diff(index_contracts_num)];
    index_contracts_num = ~eq(index_contracts_num,0);

    tref_str = index_data(:,1);
    tref = datenum(tref_str);
    o_c_price = cell2mat(index_data(:,2:3));
    open_price = o_c_price(:,1);
    close_price = o_c_price(:,2);
    %g_cum; g_jump g_inner %累计收益，跳价收益，日内收益
    %几何收益率
    g_cum = [0;log(close_price(2:end)./close_price(1:end-1))];
    g_jump = [0;log(open_price(2:end)./close_price(1:end-1))];
    g_inner = log(close_price./open_price);
    g_inner(1) = 0;

    g_cum(index_contracts_num) = 0;
    g_jump(index_contracts_num) = 0;
    g_inner(index_contracts_num) = 0;


    g_data_geo = [g_cum,g_inner,g_jump];
    %算术收益率
    g_cum_m = [0;close_price(2:end)./close_price(1:end-1)-1];
    g_jump_m = [0;open_price(2:end)./close_price(1:end-1)-1];
    g_inner_m = close_price./open_price-1;
    g_inner_m(1) = 0;

    g_cum_m(index_contracts_num) = 0;
    g_jump_m(index_contracts_num) = 0;
    g_inner_m(index_contracts_num) = 0;

    g_data_math = [g_cum_m,g_inner_m,g_jump_m];
end




function [tref,open_price,close_price] = get_etf_data(sub_index_name)
    %指数数据
    [~,~,index_data] = xlsread(sprintf('%s.csv',sub_index_name));
    index_data = index_data(5:end-1,[1,2,5]);

    tref_str = index_data(:,1);
    tref = datenum(tref_str);
    o_c_price = cell2mat(index_data(:,2:3));
    open_price = o_c_price(:,1);
    close_price = o_c_price(:,2);
    
end
