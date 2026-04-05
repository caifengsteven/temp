%{
1. 所有品种的保证金固定为 20%；
2. 仓位固定为 50%，即调仓日使用 50%的资金作为保证金买入期货合约，余下的
现金用于每日追加保证金，并按隔夜回购利率 R001 计算每日现金部分收益；
3. 交易成本：全品种按单边万分之三计算；
4. 使用复权主力合约发出交易信号，使用主力合约交易，在切换日收盘时平掉当
前仓位，在下一个主力合约上开仓，开平仓的合约价值相同

开盘买，收盘卖
增加R001收益
在ver1基础上升级
更改日期
%}
function [cash_flow,tref] = get_bac_dataV2(symbol,contract_multiplier,position_ratio,ini_cash)
    if nargin < 3
        position_ratio = 0.2;%仓位比例
    end
    if nargin < 4
        ini_cash = 10000000;%初始资金
    end
    %获取掘金主连数据
    sql_str = ['select tradingdate,open,close from futuredata.price_if_data ',...
        'where variety0=''%s'' and variety=''%s''and open>0 ',...
        'and tradingdate <= ''2017-07-31''  and tradingdate>=''2005-01-01'' order by tradingdate'];
    y_jj = fetchmysql(sprintf(sql_str,symbol{1},symbol{2}),2);

    sql_str2 = 'select tradingdate,symbol from futuredata.future_contracts_data where variety=''%s.%s'' order by tradingdate';
    sub_sql_str2 = sprintf(sql_str2,symbol{1},symbol{2});
    index_contracts = fetchmysql(sub_sql_str2,2);

    [tref,ia,ib] = intersect(y_jj(:,1),index_contracts(:,1));
    tref = datenum(tref);
    y_jj = y_jj(ia,:);
    y_jj_price_open = cell2mat(y_jj(:,2));
    y_jj_price_close = cell2mat(y_jj(:,3));
    index_contracts = index_contracts(ib,:);

    T = length(y_jj_price_open);
    fee = 3/10000;%手续费　
    %fee = 0;

    asure_ratio = 0.2; %保证金比例
    R001 = 2/100/365;%R001按照年化2%计算
    N=1e10; %换仓间隔
    %contract_multiplier = 10;%合约乘数
    cash_flow_detail_open = zeros(T,2);%开盘后，剩余流动资金,保证金
    cash_flow_detail_close=cash_flow_detail_open;%收盘后，剩余流动资金,保证金
    cash_flow = zeros(T+1,1);%记录每日资金总数
    cash_flow(1) = ini_cash;
    position_detail_open = zeros(T,2); %买入的价格 买入的手数
    %position_detail_close = position_detail_open;
    fee_flow = zeros(T,2); %开盘买入手续费，收盘卖出手续费

    for i = 1:T
        %建仓条件
        case_open1 = eq(mod(i-1,N),0);%开盘建仓
        if i > 1 %换约建仓
            case_open2 = ~strcmp(index_contracts(i,2),index_contracts(i-1,2));
        else
            case_open2 = false;
        end    
        case_open = case_open1|case_open2;
        %平仓条件
        case_close1 = eq(mod(i,N),0);%结算平仓
        if i <T
            case_close2 = ~strcmp(index_contracts(i,2),index_contracts(i+1,2));
        else
            case_close2 = true;
        end  
        case_close= case_close1|case_close2;

        if case_open %符合开仓条件
            %开盘开仓
            %1手保证金价格
            asure_grid = y_jj_price_open(i)*contract_multiplier*asure_ratio;
            %可买入的手数
            sub_share_num = floor(cash_flow(i)*position_ratio/asure_grid); 
            %记录       
            position_detail_open(i,:) = [y_jj_price_open(i),sub_share_num];
            cash_flow_detail_open(i,:) = [cash_flow(i)*(1-position_ratio),...
                cash_flow(i)*position_ratio-sub_share_num*asure_grid*fee];        
            fee_flow(i,1) = sub_share_num*asure_grid*fee;
        else
            position_detail_open(i,:) = position_detail_open(i-1,:);%record
            cash_flow_detail_open(i,:) =cash_flow_detail_open(i-1,:);%record
        end    
        %收盘统计
        %参数
        if case_open
            temp_cash_flow_detail = cash_flow_detail_open(i,:);
            sub_price = y_jj_price_open(i);
        else
            temp_cash_flow_detail = cash_flow_detail_close(i-1,:).*[1+R001,1];
            sub_price = y_jj_price_close(i-1);
        end
        %质押金变化
        temp_cash_flow_detail(2) = temp_cash_flow_detail(2)+(y_jj_price_close(i)-...
            sub_price)*contract_multiplier*position_detail_open(i,2);
        if case_close %符合平仓条件
            fee_flow(i,2) = y_jj_price_close(i)*contract_multiplier*position_detail_open(i,2)*fee;
            temp_cash_flow_detail(2) = temp_cash_flow_detail(2)-fee_flow(i,2);
            cash_flow_detail_close(i,:) = [sum(temp_cash_flow_detail),0];
            cash_flow(i+1) =  sum(temp_cash_flow_detail);
        else
            %仅仅结算        
            %保证金变化
            temp_asure_cash = sub_price*contract_multiplier*position_detail_open(i,2)*asure_ratio;
            %是否需要追加
            if temp_asure_cash>temp_cash_flow_detail(2)%追加
                cash_need = temp_asure_cash-temp_cash_flow_detail(2);
                temp_cash_flow_detail = temp_cash_flow_detail+[-cash_need,cash_need];
            end
            cash_flow_detail_close(i,:) = temp_cash_flow_detail;
            cash_flow(i+1) = sum(cash_flow_detail_close(i,:));
            if cash_flow(i+1)<=0
                cash_flow(i+1:end) = cash_flow(i+1);
                break
            end
        end
        %if any(isnan(cash_flow));keyboard;end

    end
    
    cash_flow = cash_flow(2:end);
%     figure;
%     plot(y_jj_price_close/y_jj_price_close(1))
%     hold on
%     plot(cash_flow/cash_flow(1))
    %fee_all = cumsum(sum(fee_flow,2))/cash_flow(1);
    %plot(fee_all);
end