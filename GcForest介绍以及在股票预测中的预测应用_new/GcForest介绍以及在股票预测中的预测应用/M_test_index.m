clear
close all
symbol_pool_all = {'000001','399300','000905','000016','000906'};
T_symbol_pool_all = length(symbol_pool_all);

%%{
parfor i = 1:T_symbol_pool_all
    dos_str = sprintf('python M_gcforest.py %s lo',symbol_pool_all{i});
    %sprintf(dos_str)
    dos(dos_str)
end
%}
for i = 1:T_symbol_pool_all
    [~,~,x] = xlsread(sprintf('gp_%s.csv',symbol_pool_all{i}));
    t =cellstr(datestr( datenum(x(2:end,1)),'yyyy-mm-dd'));
    x = cell2mat(x(2:end,2));
    
    r_c = cumprod(1+x);
    t_str = cellfun(@(x) [x(1:4),x(6:7),x(9:10)],t,'UniformOutput',false);

    T = length(t_str);
    h=figure;
    title_str = symbol_pool_all{i};
    plot(r_c,'-','LineWidth',2);
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str(floor(linspace(1,T,15))));
    set(gca,'XTickLabelRotation',90)    
    setpixelposition(h,[223,365,1345,420]);
    box off
    title(title_str)    
    
end