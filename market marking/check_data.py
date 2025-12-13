import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

# Read with filter for AAPL and MSFT
filters = [('ticker', 'in', ['AAPL', 'MSFT'])]
table = pq.read_table('W:/us_stock_quotes_parquet/2024/01/2024-01-02.parquet', filters=filters)
df = table.to_pandas()

print('Shape after filter:', df.shape)
print('\nAAPL sample:')
aapl = df[df['ticker'] == 'AAPL']
print(aapl.head(20))
print('\nMSFT sample:')
msft = df[df['ticker'] == 'MSFT']
print(msft.head(20))

print('\n\nAAPL rows:', len(aapl))
print('MSFT rows:', len(msft))
print('\nAAPL price range: bid', aapl['bid_price'].min(), '-', aapl['bid_price'].max())
print('AAPL price range: ask', aapl['ask_price'].min(), '-', aapl['ask_price'].max())

