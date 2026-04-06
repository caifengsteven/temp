clear

key_str = 'S43美股验证';
write_sel_excel = true;
pn_write = fullfile(pwd,'计算结果');
if write_sel_excel    
    if ~exist(pn_write,'dir')
        mkdir(pn_write)
    end
    
    xls_fn = fullfile(pn_write,sprintf('%s.xlsx',key_str));
end

write_sel_word=false;
if write_sel_word
	if ~exist(pn_write,'dir')
        mkdir(pn_write)
    end    
    obj_wd = wordcom(fullfile(pn_write,sprintf('%s.doc',key_str)));
end

t_fee = 0.5/10000;

[~,~,X] = xlsread('S43_americanstock_re.csv');
tref = X(2:end,end-1);
symbols = X(2:end,end);
var_name = X(1,2:end-2);
X = cell2mat(X(2:end,2:end-2));
X(isnan(X)) = 1;

u_symbols = unique(symbols);
T_u_symbols = length(u_symbols);
sta_re1 = cell(T_u_symbols,1);
for symbol_sel = 1:T_u_symbols
    sub_symbol = u_symbols{symbol_sel};
    sub_ind = strcmp(symbols,u_symbols(symbol_sel));
    x=X(sub_ind,:);
    t = tref(sub_ind);
    t = cellstr(datestr(datenum(t),'yyyy-mm-dd'));
    t_str = cellfun(@(x) [x(1:4),x(6:7),x(9:10)],t,'UniformOutput',false);
    T = length(t_str);
    T_type = size(x,2);
    sta_re2 = cell(T_type,1);
    for i = 1:T_type
        sub_r = x(:,i);
        sub_r(2:end) = sub_r(2:end)./sub_r(1:end-1)-1;
        sub_r(1) = 0;
        %找到时间点
        sub_r = abs(sub_r)>0;
        id = find(diff(sub_r))+1;
        x(id,i) = x(id,i)-t_fee/2;
        r_c = cumprod(x(:,i));
        title_str = sprintf('%s-%s',sub_symbol,var_name{i});
        title_str(strfind(title_str,'_')) = '-';
        if write_sel_word
            h=figure;            
            plot(r_c,'-','LineWidth',2);
            set(gca,'xlim',[0,T]);
            set(gca,'XTick',floor(linspace(1,T,15)));
            set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
            set(gca,'XTickLabelRotation',90)    
            setpixelposition(h,[223,365,1345,420]);
            box off
            title(title_str)  
            obj_wd.pasteFigure(h,title_str);  
        end
        
        [v0,v_str0] = curve_static(r_c,[],false);
        [v,v_str] = ad_trans_sta_info(v0,v_str0); 
        result2 = [v_str;v]';
        result = [{sub_symbol,title_str};result2];
        if ~eq(i,1)
            result = result(:,2);
        end
        sta_re2{i} = result;
        sprintf('%s %d-%d',key_str,i,T_type)
    end
    y = [sta_re2{:}];
    sta_re1{symbol_sel} = y;
end
y = [sta_re1{:}];
y = y';
if write_sel_word
    obj_wd.CloseWord();
end

if write_sel_excel    
    xlswrite(xls_fn,y);
end



