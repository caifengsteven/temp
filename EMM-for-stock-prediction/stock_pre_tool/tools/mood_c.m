function [mood_s] =mood_c(date_m,date_s,opinion)
%%%����ʱ���ʽ��date_m��date_s
date_s=datenum(date_s);
date_s=fix(datenum(date_s));
date_m=datenum(date_m);
mood_series=[date_m,opinion];
mood_series_s=sortrows(mood_series,1);
opinion=mood_series_s(:,2);
date_m=fix(mood_series_s(:,1));
date_stock=sort(date_s);
datepoint=unique(date_m);

% date_stock=fix(date_s);
%%%%% ͳ��������㿴��ָ��
[n,~]=size(datepoint);
total=zeros(5,n);%%%��ֵ��������,0,��ֵ��������%%
for i=1:length(opinion)
    day=date_m(i);
    j=find(datepoint==day);
    if opinion(i)>0
        total(1,j)= total(1,j)+opinion(i);
        total(2,j)=total(2,j)+1;
    elseif opinion(i)<0
         total(4,j)= total(4,j)+opinion(i);
         total(5,j)=total(5,j)+1;
       elseif opinion(i)==0
        total(3,j)= total(3,j)+1;   
    end
end
total_n=total(3,:)+total(2,:)+total(5,:);
ET=log((total(1,:)+0.0001)./(-total(4,:)+0.0001));
ET=zscore(ET);%%%��Ϊ��ֵΪ0����׼��Ϊ1�ı�׼������
total=zscore(total);%%%��Ϊ��ֵΪ0����׼��Ϊ1�ı�׼������


dw=datepoint;
day_real=[];
ET_p=[];
tn_p=[];
count=0;
lth=length(date_stock);
mood_s=zeros(lth,2);
for i=1:lth
    a=find(dw==date_stock(i));
    if (~isempty(a))
    mood_s(i,1)=ET(a);
    mood_s(i,2)=total_n(a);
    end
end
end
