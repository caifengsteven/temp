clear
obj = file_manager();

file_type = '*.zip';
pn = 'G:\dropbox\Dropbox\Dropbox\Data\Uqer_download_factor_data';
%pn1 = 'I:\data\ycz_fenbi';
pn_to = [pn,'_all_csv_files'];
fns  = get_all_files(obj,pn,file_type);
T= length(fns);
parfor i = 1:T
%for i = 1:T
    sub_fn = fns{i};
    [sub_filepath,sub_name] = fileparts(sub_fn);
    %sub_pn_to = fullfile(sub_filepath,sub_name);
    sub_pn_to = sub_filepath;
    sub_pn_to = [pn_to,sub_pn_to(length(pn)+1:end)];
    dos_unzip(sub_fn,sub_pn_to)
end