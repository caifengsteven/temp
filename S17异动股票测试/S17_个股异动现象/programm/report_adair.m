function report_adair(key_str,H,Y,info,L)
    if nargin < 5
        L = cell(size(Y));
    end
    %key_str = 'S43项目';
    file_name = sprintf('%s%s',key_str,datestr(now,'yyyy-mm-dd'));
    pn0 = fullfile(pwd,'计算结果');
    if ~exist(pn0,'dir')
        mkdir(pn0);
    end
    if exist('wordcom_bac.m','file')
        obj_wd = wordcom_bac(fullfile(pn0,sprintf('%s.doc',file_name)));
    else
        obj_wd = wordcom(fullfile(pn0,sprintf('%s.doc',file_name)));
    end
    for i = 1:length(H)
        obj_wd.pasteFigure(H(i),' ');
    end
    obj_wd.CloseWord();
    yc = Y;
    yt = info;
    %yt = [yt{:}];
    sta_re = curve_static_batch_v2(yc,yt,L);
    xlstocsv_adair(fullfile(pn0,sprintf('%s.xlsx',file_name)),sta_re) 
end