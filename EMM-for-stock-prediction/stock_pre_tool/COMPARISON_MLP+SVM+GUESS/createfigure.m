function createfigure(yvector1, E1)
%CREATEFIGURE(YVECTOR1, E1)
%  YVECTOR1:  bar yvector
%  E1:  errorbar e

%  �� MATLAB �� 08-Nov-2016 23:57:27 �Զ�����

% ���� figure
figure1 = figure;

% ���� axes
axes1 = axes('Parent',figure1,'XTickLabel',{'RAND','MLP','SVM','RNN'},...
    'XTick',[1 2 3 4],...
    'FontWeight','bold',...
    'FontSize',16);
%% ȡ�������е�ע���Ա���������� Y ��Χ
% ylim(axes1,[0.4 0.8]);
box(axes1,'on');
hold(axes1,'on');

% ���� ylabel
ylabel('accuracy');

% ���� bar
bar(yvector1,'FaceColor',[1 1 1],'BarWidth',1);
end


