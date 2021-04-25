import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


# metrics
def cal_sharpe(ts, rf=0.0):
    chgs = np.log(ts).diff()
    mu = chgs.mean()*252
    sigma = chgs.std()*np.sqrt(252)
    sharpe = (mu - rf)/(sigma)
    return namedtuple('Sharpe', 'mu, sigma, sharpe')(
                      mu, sigma, sharpe)


def cal_drawdown(ts):
    ts = np.log(ts)
    run_max = np.maximum.accumulate(ts)
    end = (run_max - ts).idxmax()
    start = (ts.loc[:end]).idxmax()
    low = ts.at[end]
    high = ts.at[start]
    mdd = np.exp(low) / np.exp(high) - 1
    points = [(start, np.exp(high)), (end, np.exp(low))]
    duration = len(ts.loc[start:end])
    return namedtuple('Drawdown',
                      'mdd, points, duration')(
                       mdd, points, duration)

# plotting
def normal_plot(ts, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    ts.plot(ax=ax)
    ax.legend()
    plt.show()
    
    
def logy_plot(ts, **kwargs):
    fig, ax = plt.subplots(1, 1, **kwargs)
    formatter = lambda x, pos: f'{np.exp(x):.2f}'
    ax.yaxis.set_major_formatter(formatter)
    np.log(ts).plot(ax=ax)
    ax.legend()
    plt.show()
