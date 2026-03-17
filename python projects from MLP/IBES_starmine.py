from datasets.ibes.data_source import *
from lager.datasources.arm import *
from lager.datasources.barra import *

ds= ARMUSDatasource()
ds1 = ARMEuroDatasource()

startdate = pd.Timestamp('20180101')
enddate = pd.Timestamp('20190630')
df_arm = ds.query(start_date = startdate, end_date = enddate, split = 'US')
df_us = ds1.query(start_date = startdate, end_date =enddate)

print(df_arm.columns.tolist())
print(df_eu.columns.tolist())

ds_est_us = IBESEstimatesINT()
stocklist['@5AH', '@1997', '@VZR', '@VPI', '@XN2']

ds_est_us = ds_est_us.get(from_date='20190101', to_date='20190630', symbols= stocklist)

ds_est_us['measure'].unique()
ds_rec_us = IBESRecommendationsUS()
