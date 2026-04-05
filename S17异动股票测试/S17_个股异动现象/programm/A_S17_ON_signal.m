%{
尾盘信号
尾盘30分钟内触发信号后开仓、收盘平仓、开盘平仓
阈值包括1%、2%、3%、4%、5%

上涨异动，文章做法是立即做空，第二天开盘平仓
下跌异动，文章做法是立即做多，第二天开盘平仓
%}
clear


cut_value = (1:5)/100;

sql_str = 'show tables from ycz_min_history';
tref = fetchmysql(sql_str,2);
del_ind = cellfun(@length,tref);
tref = tref(eq(del_ind,8));
[~,ia] = sort(tref);
tref = tref{end};

tref_num = datenum(tref,'yyyymmdd');
tref2 = cellstr(datestr(tref_num,'yyyy-mm-dd'));
tref2 = tref2{1};

%数据未缺失
sql_str1 = ['select symbol,tradingdate,close from ycz_min_history.`%s` ',...
    'where  (hour(tradingdate)>=14 and minute(tradingdate)>=30) ',...
    'or hour(tradingdate)=15  order by symbol,tradingdate'];

sub_x = fetchmysql(sprintf(sql_str1,tref),2);
tickers = unique(sub_x(:,1));

T = length(tickers);
re1 = cell(T,1);
re2 = cell(T,1);
parfor i = 1:T

    sub_sub_x_a = sub_x(strcmp(sub_x(:,1),tickers(i)),:);
    temp_v3 = sub_sub_x_a{end,end}; %收盘价
    sub_sub_x = cell2mat(sub_sub_x_a(:,3));
    sub_sub_r = [0;sub_sub_x(2:end)./sub_sub_x(1:end-1)-1];   
    sub_sub_r(isinf(sub_sub_r)) = 0; %del inf
    for k = 1:5
        sub_ind = find(sub_sub_r>cut_value(k),1);
        if ~isempty(sub_ind)
            %代码，时间，前收，现收，当天收盘价，阈值
            sub_sub_re = {tickers{i},sub_sub_x_a{sub_ind,2},...
                sub_sub_x(sub_ind-1),sub_sub_x(sub_ind),temp_v3,k};
            re1{i} = sub_sub_re';
        end
    end

    %信号2
    for k = 1:5
        sub_ind = find(sub_sub_r<-cut_value(k),1);
        if ~isempty(sub_ind)
            %代码，时间，前收，现收，收盘价，时间间隔，第二日开盘价
            sub_sub_re = {tickers{i},sub_sub_x_a{sub_ind,2},...
                sub_sub_x(sub_ind-1),sub_sub_x(sub_ind),temp_v3,-k};
            re2{i} = sub_sub_re';
        end
    end
    sprintf('%d-%d %d-%d',i,T)
end
re1=deal_data1(re1);
re2 = deal_data1(re2);
var_names1 = {'股票代码','发生时间','触发前价','触发时价','收盘价','上涨异动信号等级（当日做空次日平仓）'};
var_names2 = {'股票代码','发生时间','触发前价','触发时价','收盘价','下跌异动信号等级（当日做多次日平仓）'};

m = max([size(re1,1),size(re2,1)]);
re = cell(m,size(re1,2)*2);
re(1:size(re1,1),1:size(re1,2)) = re1;
re(1:size(re2,1),1+size(re1,2):end) = re2;

t_str=sprintf('S17个股异动信号%s',tref2);
gui_result(re,t_str,[var_names1,var_names2])
    
re3 = [[var_names1,var_names2];re];
re3 = cell2table(re3);
writetable(re3,sprintf('%s.csv',t_str));
