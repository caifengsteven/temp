%M_check_missing_data

[~,~,info] = xlsread('data.xlsx','sheet2');
info = cellfun(@(x) x(1:end-3),info,'UniformOutput',false);

fns= dir(fullfile('data','*.txt'));
fns = {fns.name}';
fns = cellfun(@(x) x(end-9:end-4),fns,'UniformOutput',false);

setdiff(info,fns)