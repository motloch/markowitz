"""
Plot expected returns/volatility of a linear combination of several stocks. Also show the
efficient frontier a la Markowitz.

From https://towardsdatascience.com/python-markowitz-optimization-b5e1623060f5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from scipy.optimize import minimize
from os.path import exists

assets = ['AAPL', 'CSCO', 'IBM', 'AMZN']
filename = 'pf_data2.csv'

# Load data, or if they do not exist, download them from the web
if exists(filename):
    pf_data = pd.read_csv(filename, index_col = 0)
else:
    pf_data = pd.DataFrame()
    for x in assets:
        pf_data[x] = wb.DataReader(x, data_source='yahoo', start = '2015-1-1')['Adj Close']
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
portfolio_sharpe = []

for x in range(10000):
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)

    ret = np.sum(weights * means)
    var = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    portfolio_returns.append(ret)
    portfolio_volatilities.append(var)
    portfolio_sharpe.append(ret/var)

portfolios = pd.DataFrame({'Return': portfolio_returns, 
                            'Volatility': portfolio_volatilities, 
                            'Sharpe': portfolio_sharpe
                            })

# get the efficient boundary
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(means * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sr = ret/vol

    return np.array([ret, vol, sr])

def neg_sharpe(weights):
    return -get_ret_vol_sr(weights)[2]

def check_sum(weights):
    return np.sum(weights)-1

cons = ({'type': 'eq', 'fun':check_sum})
bounds = ((0,1), (0,1), (0,1), (0,1))
init_guess = [0.25, 0.25, 0.25, 0.25]

opt_results = minimize(neg_sharpe, init_guess, method = 'SLSQP', bounds = bounds, constraints=cons)

frontier_y = np.linspace(2e-4, 8.7e-4, 400)

def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]

frontier_x = []

for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun':lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = minimize(minimize_volatility, init_guess, method='SLSQP', bounds = bounds,
    constraints = cons, tol = 1e-8)
    frontier_x.append(result['fun'])

# Plot the results
portfolios.plot(x='Volatility', y='Return', c='Sharpe', kind='scatter', cmap='viridis')
plt.xlabel('Expected Daily Volatility')
plt.ylabel('Expected Daily Return')
plt.plot(frontier_x,frontier_y, 'r--', linewidth=3)
plt.savefig('markowitz2.png')
