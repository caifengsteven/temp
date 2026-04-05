clear
load F22_IF
tref = F(:,1);
T = 3;

dns = {'IF','IH','IC'};
method_name = {'复合策略1','复合策略2','复合策略3'};

tb_x =[];
tb_str = [];
tb_y = [];
for i = 1:T
    temp = load(sprintf('final_re_f%d.mat',i));
    temp = temp.Y_re;
    for j = 1:size(temp,2)
        [v,v_str,sta_val] = curve_static(temp(:,j));
        v([1:5,12:13]) = v([1:5,12:13]) * 100;
        if isempty(tb_x)
            tb_x = v';
            tb_str = {sprintf('%s-%s',method_name{i},dns{j})};
            tb_y = v_str';
        else
            tb_x = cat(2,tb_x,v');
            tb_str = cat(2,tb_str,{sprintf('%s-%s',method_name{i},dns{j})});
        end
    end
    if eq(i,1)
        x = temp;
    else
        x = cat(2,x,temp);
    end
    
    
    
end
tb_str = [{[]},tb_str];
tb = [tb_str;tb_y,num2cell(tb_x)];
%%%%%%%%%%%%%%%%%%%%%%%%%
T = length(tref);
for i = 1:3
    
    figure
    plot(x(:,i:3:end)*100,'LineWidth',2);
    legend(method_name,'NumColumns',3,'Location','northwest')
    set(gca,'xlim',[0,T]);
    set(gca,'XTick',floor(linspace(1,T,15)));
    t_str = tref(floor(linspace(1,T,15)));
    set(gca,'XTickLabel',t_str);
    set(gca,'XTickLabelRotation',90)
    title(dns{i})
    setpixelposition(gcf,[223,365,1345,420]);
    box off
        
end
%打印结果
tb = tb';
[m,n] = size(tb);
for i = 1:m
    temp = tb(i,:);
    for j = 1:n
        temp_temp = temp{j};
        if isempty(temp_temp)
            fprintf('\t')
        elseif ischar(temp_temp)
            fprintf('%s\t',temp_temp)
        else
            fprintf('%0.4f\t',temp_temp)
        end
    end
    fprintf('\n')
end