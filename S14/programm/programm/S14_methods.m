classdef S14_methods < handle
    methods(Static)
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
    end
end