clear
fns = {'re_hs_300.csv';'re_hs_500.csv';'re_hs_a.csv'};

T = length(fns);
for i =1:T
    [~,~,x] = xlsread(fns{i});
    tref = x(2:end,1);
    y = cell2mat(x(2:end,2:end));
    legend_str = x(1,2:end);
    for j = 1:length(legend_str)
        sub_str = legend_str{j};
        sub_str(strfind(sub_str,'_')) = '-';
        legend_str{j} = sub_str;
    end
    title_str = fns{i};
    title_str(strfind(title_str,'_')) = '-';
    title_str = strsplit(title_str,'.');
    title_str = title_str{1};
    draw_figure(tref,y,title_str,legend_str)
end


function draw_figure(tref,r_day,title_str,legend_str)
    r_c = cumprod(1+r_day);
    t_str = cellstr(datestr(datenum(tref),'yyyymmdd'));

    T = length(t_str);
    h1=figure;

    plot(r_c(:,1:end),'-','LineWidth',2);
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
    set(gca,'XTickLabelRotation',90)    
    setpixelposition(h1,[223,365,1345,420]);
    box off
    legend(legend_str,'NumColumns',length(legend_str),'Location','best');
    title(sprintf('%s ·Ö×é¾»Öµ',title_str));

end