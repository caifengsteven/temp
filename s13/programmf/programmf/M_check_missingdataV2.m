%M_check_missing_data

[~,~,info] = xlsread('data.xlsx','sheet3');
info_str = cell(size(info,1),1);
for i = 1:length(info)
    info_str{i} = sprintf('%s%s%s.txt',info{i,2},info{i,3},info{i,1}(1:6));
end


fns= dir(fullfile('data','*.txt'));
fns = {fns.name}';


setdiff(info_str,fns)