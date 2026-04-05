%从yuqer获取数据方法合集
classdef yq_methods< handle
    methods(Static)
        %输入t1 起点  t2 终点
        function tref = get_tradingdate(t1,t2)
            if nargin < 1
                t1 = [];
            end
            if nargin < 2
                t2=[];
            end
            
            if isempty(t1)
                t1 = '1980-01-01';
            end
            if isempty(t2)
                t2 = '2999-01-01';
            end
            
            sql_str = ['select tradingdate from yuqerdata.yq_tradingdate ',...
                'where tradingdate >= ''%s'' and tradingdate<=''%s'' order by tradingdate ;'];
            sql_str = sprintf(sql_str,t1,t2);
            tref = fetchmysql(sql_str,2);
        end
        %获取合并利润表数据
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_HeBingLiRun(var_name)
            sql_str1 = ['select ticker,publishdate,enddate,fiscalPeriod,',...
                'reportType,%s from yuqerdata.nincome '];
            sql_str2 = [' where enddate>''2008-01-01'' and enddate=endDaterep ',...
                'and mergedFlag = 1 and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'and reporttype in (''Q1'',''S1'',''CQ3'',''A'') order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取合并现金流量表
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_HeBingXianJinLiu(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,fiscalPeriod,',...
                'reportType,%s from yuqerdata.yq_FdmtCFGetAll '];
            sql_str2 = [' where enddate>''2008-01-01'' and enddate=endDaterep ',...
                'and mergedFlag = 1 and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'and reporttype in (''Q1'',''S1'',''CQ3'',''A'') order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取财务指标.盈利能力表数据
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_YingLiNengLi(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiRtnPitGet '];
            sql_str2 = [' where enddate>''2008-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获得财务指标 衍生FdmtDerPitGet
        %symbol，发布日期，会计截至日期，时间长度，类型，数值
        function x = get_CaiWu_yansheng(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtDerPitGet '];
            sql_str2 = [' where enddate>''2008-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取财务指标.单股
        %symbol，发布日期，会计截至日期，数值
        function x = get_CaiWu_DanGu(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiPSPitGet '];
            sql_str2 = [' where enddate>''2008-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取财务指标.运营能力
        %symbol，发布日期，会计截至日期，数值
        function x = get_CaiWu_YunYingNengLi(var_name)
            sql_str1 = ['select symbol,publishdate,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiTrnovrPitGet '];
            sql_str2 = [' where enddate>''2008-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取单季度财务指标 这个表没有发布日期，需要和别的表联合查询
        %symbol，发布日期，会计截至日期，数值
        function x = get_CaiWu_DanJiDu(var_name)
            sql_str1 = ['select symbol,enddate,',...
                '%s from yuqerdata.yq_FdmtIndiQGet '];
            sql_str2 = [' where enddate>''2008-01-01'' ',...
                'and (secID like ''0%'' or secID like ''60%'' or secID like ''30%'')  ',... 
                'order by secID,enddate desc'];
            sql_str1 = sprintf(sql_str1,var_name);
            sql_str = [sql_str1,sql_str2];
            x = fetchmysql(sql_str,2);
        end
        %获取某个时间点市值
        function x = get_market_value(t)
            %当前股票市值
            sql_str_f1 = 'select symbol,log(marketValue) from yuqerdata.yq_dayprice where tradedate = ''%s'' and marketValue is not null and marketValue !=0 order by symbol';
            x = fetchmysql(sprintf(sql_str_f1,t),2);
        end
        %获取某个时间点申万一级行业分类
        function x = get_industry_class(t)
            str_str1 = ['select ticker,industryID1 from yuqerdata.yq_industry where ',...
                'industryVersionCD=''010303'' and intodate <= ''%s'' and ',...
                '(outDate>''%s'' or outDate is null)'];
            sql_str = sprintf(str_str1,t,t);
            x = fetchmysql(sql_str,2);
        end
        %获取交易日历
        function x= get_trading_date()
            sql_str = 'select tradingdate from yuqerdata.yq_tradingdate order by tradingdate';
            x = fetchmysql(sql_str,2);
        end
        %ST,*ST,PT code
        function x = get_stpt_symbol(t1)
            sql_str= ['select distinct(symbol) from yuqerdata.st_info where ',...
                'tradingdate = ''%s'''];
            x = fetchmysql(sprintf(sql_str,t1),2);
        end
        %获取上市日期限制的symbol
        function x = get_time_cut_symbol(t0)
            sql_str = ['select symbol from yuqerdata.stock_basic_info where listdate<''%s'' ',...
                'and listdate is not null'];
            x = fetchmysql(sprintf(sql_str,t0),2);
        end
        %获取某日的可用symbol
        function x = get_stop_symbol(t0)
            sql_str = ['select distinct(symbol) from yuqerdata.yq_stop_run_data where ',...
                'haltbegintime <= ''%s 09:30:00'' and haltEndTime>=''%s''' ];
            x = fetchmysql(sprintf(sql_str,t0,t0),2);
        end
        
    end
    
    
end