%格点搜索 S43_cf_0.00_0.00_1.csv
clear
pn = '优化结果';

para1 = 0:0.05:0.3;
para2 = 0:0.05:0.5;

X = nan(length(para1),length(para2));

for i = 1:length(para1)
    for j = 1:length(para2)
        sub_fn = sprintf('S43_cf_%0.2f_%0.2f_1.csv',para1(i),para2(j));
        [~,~,x] = xlsread(fullfile(pn,sub_fn));
        
        if size(x,1)<1000
            continue
        end
        
        symbol = x(2:end,end);
        symbol_u = unique(symbol);
        sub_y = cell2mat(x(2:end,5));
        
        sub_re = nan(size(symbol_u));
        for symbol_sel = 1:length(symbol_u)
            sub_sub_y = sub_y(strcmp(symbol,symbol_u(symbol_sel)));
            sub_sub_y = cumprod(sub_sub_y);
            
            %sharp
            v = curve_static(sub_sub_y);
            if ~isnan(v(9))
                sub_re(symbol_sel) = v(9);
            end
            
        end
        sub_re = mean(sub_re(~isnan(sub_re)));
        X(i,j) = sub_re;
    end
    sprintf('%d-%d',i,j)
    
    
end


