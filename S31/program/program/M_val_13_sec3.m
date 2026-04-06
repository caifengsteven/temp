%周内效应
clear
symbol = {'000016','399300','000905'};
symbol_info = {'上证50','沪深300','中正500'};
sql_str = ['select tradedate,weekday(tradedate),chgPct from yuqerdata.yq_index where ',...
    'symbol = ''%s'' and tradedate>=''2014-05-05'' and tradedate<=''2019-12-31'' order by tradedate '];

re = cell(3,1);
figure
for i = 1:3
    sub_x = fetchmysql(sprintf(sql_str,symbol{i}),2);
    sub_y0 = cell2mat(sub_x(:,2:end));
    sub_y0(eq(sub_y0(:,1),3),:) = [];
    %sub_y0 = [];
    y_dir = ones(size(sub_y0(:,1)));
    y_dir(eq(sub_y0(:,1),2)|eq(sub_y0(:,1),3))=-1;
    
    T = length(y_dir);
    id =0;
    y_re = zeros(T,2);
    y_re_count = 1;
    t_ind = zeros(T,1);
    for j = 1:T
        if eq(j,1)
            id = 1;
            sub_y = 1*(1+sub_y0(j,2));
        else
            if ~eq(y_dir(j),y_dir(j-1))
                t_ind(j) = 1;
                t = sub_y-1;
                if t>0
                    temp = 1;
                else
                    temp = -1;
                end
                y_re(y_re_count,:) = [y_dir(j-1),temp];
                y_re_count = y_re_count+1;
                id = id + 1;
                sub_y = 1*(1+sub_y0(j,2));
            else
                sub_y = sub_y*(1+sub_y0(j,2));
            end
        end
        
    end
    y_re = y_re(1:y_re_count-1,:);
    y2 = y_re(:,1);
    y2(eq(y_re(:,1),y_re(:,2))) = 1;
    y2(~eq(y_re(:,1),y_re(:,2))) = -1;
    plot(cumsum(y2),'LineWidth',3)
    if eq(i,1)        
        hold on
    end
end

t_str = sub_x(eq(t_ind,1),1);
T=length(t_str);
set(gca,'xlim',[0,T]);
set(gca,'XTick',floor(linspace(1,T,15)));

set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
set(gca,'XTickLabelRotation',90)    
setpixelposition(gcf,[223,365,1345,420]);
legend(symbol_info,'NumColumns',3,'Location','best');
box off