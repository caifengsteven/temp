clear
%pn1 = 'D:\works2018\SOME\tempdatasets';
pn1 = '/home/adair/workspool/YCZ_fenbi/';
date_num = 2013;

fn_re = cell(length(date_num)*12*22,2);
fn_num = 0;

for i = 1:length(date_num)
    pn2 = fullfile(pn1,num2str(date_num(i)));
    %getdir
    sub_pn3 = get_pns(pn2);
    for j = 1:length(sub_pn3)
        pn3 = fullfile(pn2,sub_pn3{j});
        sub_fns = dir(fullfile(pn3,'*.zip'));
        sub_fns = {sub_fns.name};
        for k = 1:length(sub_fns)
            sub_pn = strsplit(sub_fns{k},'.');
            fn_num = fn_num+1;
            fn_re(fn_num,:) = {fullfile(pn3,sub_pn{1}),fullfile(pn3,sub_fns{k})};
        end
        %record        
    end
  
end
fn_re = fn_re(1:fn_num,:);

T = size(fn_re,1);
parfor i = 1:T
    sub_fn_re = fn_re(i,:);
    system(['unzip -o -d ',sub_fn_re{1},' ',sub_fn_re{2}])
    %dos_unzip(sub_fn_re{2},sub_fn_re{1})
    sprintf('%d-%d',i,T)
end



function x = get_pns(pn)
x = dir(pn);
ind = [x.isdir];
x = {x.name};
x = x(ind);

del_p = {'.','..'};
[~,ia] = intersect(x,del_p);
x(ia) = [];
x = [x,pn];

end