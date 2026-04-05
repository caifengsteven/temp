clear
parfor i = 0:5
    dos(['python M_zhubi_factor.py  ',num2str(i)])
end