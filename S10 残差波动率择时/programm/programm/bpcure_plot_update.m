%t
%x
function bpcure_plot_update(t,x,str,lstr,sel)
%3/1
if nargin < 5
    sel = 2;%Á½ÖÖÍ¼¶¼»­
end
if nargin < 4
    lstr = [];
end
if nargin < 3
    str = [];
end
if nargin < 2
    x = t(:,2);
    t = t(:,1); 
end

if eq(sel,2)
%     hold on
%     ylims1 = [min(min(x)),max(max(x))];
%     pic_lim = [-diff(ylims1)/100,diff(ylims1)/100];
%     ylims1 = ylims1+pic_lim;
%     ylims1(1) = ylims1(1)-diff(ylims1)/3;
%     ylim(ylims1)

    v = getdrawdown(x)*100;
    %subplot(1,2,2);
    yyaxis right
    %bar(t,v)
    %plot(t,v);
    %myplot(t,v);
    obj_L = mybar(t,v,[0.5,0.5,0.5]);
%    lims = axis(gca);
%     plot(lims(1:2),[0,0],'linewidth',2)
    datetick('x','yymm');

%     ylims1 = [min(min(v)),max(max(v))];
%     pic_lim = [-diff(ylims1)/100,diff(ylims1)/100];
%     ylims1 = ylims1+pic_lim;
%     ylims1(2) = ylims1(2)+3*diff(ylims1);
%     ylim(ylims1)
    xlim([min(t),max(t)]+[-1,1]);
    
    if  ~isempty(lstr)
        legend(obj_L,lstr,'location','best');
    end
    ylabel('drawdown');
    ah1=gca;
    ah1.YColor=[0.5,0.5,0.5];
end

%subplot(1,2,1)
x = bsxfun(@rdivide,x*100,x(1,:));
if eq(sel,2)
    yyaxis left
end
obj_L2 = myplot(t,x);
if ~isempty(str)
    title(str)
end
%legend({'ref',strjoin(indicatorName0,'\r\n')},'location','best')
datetick('x','yymm');
if ~eq(sel,2)&&~isempty(lstr)
    legend(obj_L2,lstr,'location','best');
end
xlim([min(t),max(t)]+[-1,1]);
ylabel('bac curve');
set(gca,'linewidth',2);
ah2=gca;
    ah2.YColor=[0.639215686274510,0.0784313725490196,0.180392156862745];
    
    %keyboard
end

function obj_L=myplot(x,y)
    T = size(y,2);
    obj_L = zeros(T,1);
    %C = linspecer(T);
    hold on
    for i = 1:T
        obj_L(i) = plot(x,y(:,i),'-','linewidth',2,'Marker','none','color',[0.639215686274510,0.0784313725490196,0.180392156862745]);
    end

end

function obj_L=mybar(x,y,c_val)
%     T = size(y,2);
%     obj_L = zeros(T,1);
%     C = linspecer(T);
%     hold on
%     for i = 1:T
%         obj_L(i) = plot(x,y(:,i),'-','color',C(i,:),'linewidth',2,'Marker','none');
%     end
    %obj_L = bar(x,y,'FaceAlpha',1,'FaceColor',c_val,'EdgeColor','none');
    obj_L = area(x,y,'FaceColor',c_val,'EdgeColor','none','FaceAlpha',0.6);
end

function v = getdrawdown(x)
    v = zeros(size(x));
    for i = 1:size(x,2)
        v(:,i) = x(:,i)./cummax(x(:,i))-1;
    end
end

% function a_setylim()
% 
% end