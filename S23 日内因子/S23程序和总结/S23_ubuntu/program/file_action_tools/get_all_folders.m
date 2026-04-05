%µ›πÈ—∞’“À˘”–¬∑æ∂
function x = get_all_folders(pn,x)
    sub_pns_full = find_sub_dirs(pn);
    x=cat(1,x,sub_pns_full);
    T = length(sub_pns_full);
    for i = 1:T
        x = get_all_folders(sub_pns_full{i},x);
    end

end


function [sub_pns_full,sub_pns_name] = find_sub_dirs(pn)
    x = dir(pn);
    ind = [x.isdir];
    sub_pns_name = {x.name};
    sub_pns_father = {x.folder};
    sub_pns_name = sub_pns_name(ind);
    sub_pns_father = sub_pns_father(ind);
    
    del_p = {'.','..'};
    [~,ia] = intersect(sub_pns_name,del_p);
    sub_pns_name(ia) = [];
    
    T= length(sub_pns_name);
    sub_pns_full = cell(T,1);
    for i = 1:T
        sub_pns_full{i} = fullfile(sub_pns_father{i},sub_pns_name{i});
    end
end