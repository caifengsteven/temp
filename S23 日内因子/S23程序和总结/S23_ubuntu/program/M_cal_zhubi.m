clear
parfor i = 0:7
    dos(['python M_zhubi_factor.py  ',num2str(i)])
end