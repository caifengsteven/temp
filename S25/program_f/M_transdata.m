clear
datapool = {'rb','L','AL','RU','RM','J','I','HC'};
for i = 1:length(datapool)
    key_w = datapool{i};
    fn0 = sprintf('%s_factor_data.csv',key_w);
    fnt = sprintf('%s_data_update.mat',key_w);
    [~,~,x] = xlsread(fn0);

    codenum = cellfun(@(x) str2double(x(3:end)),x(2:end,2));
    tref = x(2:end,3);
    t = datenum(x(2:end,3));
    openprice = cell2mat(x(2:end,4));
    closeprice = cell2mat(x(2:end,5));
    f = cell2mat(x(2:end,6));

    save(fnt,'t','tref','openprice','closeprice','f','codenum');
    sprintf('Complete: %s',key_w)
end