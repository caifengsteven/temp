clear
datapool = {'rb','L','AL','RU','RM','J','I','HC'};
for i = 1:length(datapool)
    key_w = datapool{i};
    load( sprintf('%s_data_update.mat',key_w),'t','tref','closeprice','f','codenum');

    ind = t>=datenum(2011,2,1);
    t = t(ind);
    closeprice= closeprice(ind);
    f = f(ind);
    tref = tref(ind);
    subplot(4,2,i)
    yyaxis left
    plot(t,closeprice,'LineWidth',2)
    datetick('x','yyyy');
    %set(gca,'ylim',[0,6000]);
    ylabel('closeprice')
    yyaxis right 
    plot(t,f,'LineWidth',2)
    datetick('x','yyyy');
    %set(gca,'ylim',[-6000,0]);
    ylabel('factor')
    setpixelposition(gcf,[358,508,1139,420]);
    title(datapool{i})
end