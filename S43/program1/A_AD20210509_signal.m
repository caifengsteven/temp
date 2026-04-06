%补充程序结果
%延迟执行信号
%forex_day csi 5day
clear
close all
key_str = sprintf('S43 双底策略指数成分股组合%s',datestr(now,'yyyymmdd'));
write_sel = false;

index0 = 'HK';
para = containers.Map({'US','HK','forex-day'},[15,15,5]);

if write_sel
    pn_write = fullfile(pwd,'计算结果');
    if ~exist(pn_write,'dir')
        mkdir(pn_write)
    end
    obj_wd = wordcom(fullfile(pn_write,sprintf('%s.doc',key_str)));
    xls_fn = fullfile(pn_write,sprintf('%s.xlsx',key_str));
    index_com_fn = fullfile(pn_write,sprintf('%s成分股.mat',key_str));
end

sql_str = sprintf('select ticker,tradeDate,r_%d-1  from S37.s43_addre where index0="%s" order by tradeDate',para(index0),index0);
x = fetchmysql(sql_str,2);
x = cell2table(x,'VariableNames',{'s','t','v'});
ind = unstack(x,'v','s');
ind.Properties.RowNames = ind.t;
ind.t = [];
x = table2array(ind);
x(isnan(x)) = 0;
x(x~=0) = 1;
signal = array2table(x,'VariableNames',ind.Properties.VariableNames,'RowNames',ind.Properties.RowNames);
writetable(ind,sprintf('signal_%s.csv',index0),'WriteRowNames',true)