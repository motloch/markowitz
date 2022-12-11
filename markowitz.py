"""
Plot expected returns/volatility of a linear combination of two stocks

From https://medium.com/@zeng.simonl/the-efficient-frontier-in-python-a1bc9496a0a1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from os.path import exists

assets = ['TWTR', 'AMD']
filename = 'pf_data.csv'

# Load data, or if they do not exist, download them from the web
if exists(filename):
    pf_data = pd.read_csv(filename, index_col = 0)
else:
    pf_data = pd.DataFrame()
    for x in assets:
        pf_data[x] = wb.DataReader(x, 
                                    data_source='yahoo', 
                                    start = '2015-1-1', 
                                    end = '2020-9-10'
                                )['Adj Close']
    pf_data.to_csv(filename)

# Plot the price evolution
(pf_data / pf_data.iloc[0]*100).plot(figsize = (10,5))
plt.show()

# Statistics of the log returns
log_returns = np.log(pf_data / pf_data.shift(1))
means = log_returns.mean()
cov = log_returns.cov()

# Simulate random portfolios
portfolio_returns = []
portfolio_volatilities = []

for x in range(1000):
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)

    ret = np.sum(weights * means)
    var = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    portfolio_returns.append(ret)
    portfolio_volatilities.append(var)

portfolios = pd.DataFrame({'Return': portfolio_returns, 'Volatility': portfolio_volatilities})

# Plot the results
portfolios.plot(x='Volatility', y='Return', kind='scatter')
plt.xlabel('Expected Daily Volatility')
plt.ylabel('Expected Daily Return')
plt.text(0.0335, 1e-4, 'TWTR')
plt.text(0.039, 2.1e-3, 'AMD')
plt.show()
