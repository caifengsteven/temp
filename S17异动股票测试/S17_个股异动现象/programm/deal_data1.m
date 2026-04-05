function re1=deal_data1(re1)

    re1 = [re1{:}]';

    [~,ia] = sort(abs(cell2mat(re1(:,end))),'descend');

    re1 = re1(ia,:);
end