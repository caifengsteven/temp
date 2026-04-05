clear
x = fetchmysql('select f_val from s22.s22_factor_apb_month');
histogram(x,'Normalization' ,'probability')

sprintf('mean %0.2f %%', mean(x)*100)
sprintf('median %0.2f %%', median(x)*100)
sprintf('<0 ratio: %0.2f %%', sum(x<0)/length(x)*100)
