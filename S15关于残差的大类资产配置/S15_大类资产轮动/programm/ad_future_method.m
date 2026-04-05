%获取期货参数方法合集
%mysql
classdef ad_future_method < handle
    methods(Static)
        function sql_str = get_future_basic_info_yq(symbol)
            %上市日期，保证金比例，合约乘数，最小变动单位，最后交易日
            sql_str = ['select listDate,tradeMarginRatio,contMultNum,minChgPriceNum,',...
                'lastTradeDate from futuredata.yuqer_fushare_info where contractObject=''%s'';'];
            sql_str = sprintf(sql_str,symbol);
        end
        function sql_str = get_future_data_yq(symbol)
            %交易日期，合约代码，开，手，结
            sql_str = ['select tradedate,ticker,openprice,closeprice,settleprice from ',...
                'futuredata.yuqer_fusharedata where contractobject=''%s'' ',...
                'and contractMark=''L0'''];
            sql_str = sprintf(sql_str,symbol);
        end
        function [symbols,list_date]=get_future_listdate_yq()
            sql_str = ['select listdate,contractObject from futuredata.yuqer_fushare_info ',...
                'order by contractObject,listdate'];
            x = fetchmysql(sql_str,2);
            [symbols,ia] = unique(x(:,2));
            list_date = x(ia,1);
        end
    end
end