classdef file_manager < handle
    methods 
        %获取所有 特定 文件
        function x = get_all_files(obj,pn,file_type)
            pns = obj.get_all_folders(pn);
            x = [];
            for i = 1:length(pns)
                sub_pn = pns{i};
                sub_files = dir(fullfile(sub_pn,file_type));
                if ~isempty(sub_files)
                    temp = {sub_files.name};
                    temp = cellfun(@(x) fullfile(sub_pn,x),temp,'UniformOutput',false);
                    x = cat(1,x,temp');
                end
            end
        end
        %递归寻找所有路径
        function x = get_all_folders(obj,pn,x)
            if nargin < 3
                x = {pn};
            end
            sub_pns_full = obj.find_sub_dirs(pn);
            x=cat(1,x,sub_pns_full);
            T = length(sub_pns_full);
            for i = 1:T
                x = obj.get_all_folders(sub_pns_full{i},x);
            end

        end
    end
    methods(Static)
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
    end
end