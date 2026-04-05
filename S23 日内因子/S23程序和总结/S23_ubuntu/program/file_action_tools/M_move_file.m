%M_move_file
%find all files
clear


pn = 'D:\datasets\YCZ\fenbishuju';
x = get_all_folders(pn,{pn});

T = length(x);
parfor i = 1:T
    sub_pn = x{i};
    t = dir(fullfile(sub_pn,'*.csv'));
    if ~isempty(t)
        sub_pn_to = [pn,'_all_csv_files',+sub_pn(length(pn)+1:end)];
        
        movefile(sub_pn,sub_pn_to,'f');
        
    end
    sprintf('%d-%d',i,T)
    
end
