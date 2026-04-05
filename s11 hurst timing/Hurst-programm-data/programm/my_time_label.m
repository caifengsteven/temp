function my_time_label(ah,tref)

set(ah,'XTickLabelRotation',90);
set(ah,'XTick',linspace(tref(1),tref(end),20));
datetick('x','yyyymmdd','keepticks');