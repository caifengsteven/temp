import operator
import random

import numpy as np
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
# from scoop import futures # 多线程
from scipy import fftpack

# %% 读取数据
data = pd.read_csv('./data/hs_300.csv', index_col=0).dropna()
df_x = data[['closeIndex', 'openIndex', 'turnoverVol', 'lowestIndex', 'highestIndex', 'CHGPct']]
df_stardar_x = df_x
df_y = data['CHGPct'].shift(-1)
df_x.index, df_y.index = data['tradeDate'], data['tradeDate']
df_train_x = df_x.loc['2014': '2020', :]
df_train_y = df_y.loc[df_train_x.index]

# %%
# 定义一般函数
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def log_func(x):
    x_ = x.copy()
    x_[x_ < 0] = 1e-5
    return np.log(x_)


def hilbert(x):
    index = x.index
    return pd.Series(fftpack.hilbert(x), index=index)


def two_mean(x, y):
    return (x + y) / 2


def three_mean(x, y, z):
    return (x + y + z) / 3


def if_then_esle(a, b, c, d):
    sig1 = a < b
    c_ = d.copy()
    c_[sig1] = c[sig1]
    return c_


def if_then_esle_0(a, b, c):
    sig1 = a < b
    c_ = c.copy()
    c_[sig1] = 0
    return c_


# 创建计算集合
pset = gp.PrimitiveSet("MAIN", 6)
for i, col in enumerate(df_x.columns):
    exec('pset.renameArguments(ARG%d="%s")' % (i, col))

# 添加函数
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(log_func, 1, name='log')
pset.addPrimitive(protectedDiv, 2, name='div')
pset.addPrimitive(operator.abs, 1)
pset.addPrimitive(np.sign, 1)
pset.addPrimitive(hilbert, 1)
pset.addPrimitive(lambda x: x, 1, 'self')
pset.addPrimitive(two_mean, 2)
pset.addPrimitive(three_mean, 3)
pset.addPrimitive(if_then_esle, 4)
pset.addPrimitive(if_then_esle_0, 3)


# 单个时间函数
def ts_mean(x, n):
    return x.rolling(n).mean()


def ts_max(x, n):
    return x.rolling(n).max()


def ts_std(x, n):
    return x.rolling(n).std()


def ts_min(x, n):
    return x.rolling(n).min()


def ts_delay(x, n):
    return x.shift(n)


def ts_prod(x, n):
    return x.rolling(n).apply(lambda x: x.prod())


def ts_delta(x, n):
    return x / (x.shift(n) + 0.0001) - 1


def ts_ema(x, n):
    return x.ewm(n).mean()


def ts_dema(x, n):
    ema = ts_ema(x, n)
    err = ts_ema(x - ema, n)
    dema = ema + err
    return dema


def ts_kama(x, n):
    abs_diff = ((x - x.shift(1)).abs()).rolling(n).sum()
    err = x - x.shift(n)
    er = err / abs_diff
    return ts_ema(er, n)


for n in [5, 10, 15, 20]:
    pset.addPrimitive(lambda x: ts_mean(x, n), 1, name='ts_mean_%s' % n)
    pset.addPrimitive(lambda x: ts_std(x, n), 1, name='ts_std_%s' % n)
    pset.addPrimitive(lambda x: ts_max(x, n), 1, name='ts_max_%s' % n)
    pset.addPrimitive(lambda x: ts_min(x, n), 1, name='ts_min_%s' % n)
    pset.addPrimitive(lambda x: ts_delay(x, n), 1, name='ts_delay_%s' % n)
    pset.addPrimitive(lambda x: ts_prod(x, n), 1, name='ts_prod_%s' % n)
    pset.addPrimitive(lambda x: ts_delta(x, n), 1, name='ts_delta_%s' % n)
    pset.addPrimitive(lambda x: ts_ema(x, n), 1, name='ts_ema_%s' % n)
    pset.addPrimitive(lambda x: ts_dema(x, n), 1, name='ts_dema_%s' % n)
    pset.addPrimitive(lambda x: ts_kama(x, n), 1, name='ts_kema_%s' % n)


# 交互时间函数
def ts_corr(x, y, n):
    return x.rolling(n).corr(y.rolling(n))


def ts_cov(x, y, n):
    return x.rolling(n).cov(y.rolling(n))


def ts_beta(x, y, n):
    rolling_cov = x.rolling(n).cov(y.rolling(n))
    rolling_var = x.rolling(n).var()
    return rolling_cov / rolling_var


def ts_emal(x, y, n):
    x_ = ts_ema(x, n)
    y_ = ts_ema(y, n)
    emal_beta = ts_beta(x_, y_, n)
    return emal_beta


for n in [5, 10, 15, 20]:
    pset.addPrimitive(lambda x, y: ts_corr(x, y, n), 2, name='ts_corr_%s' % n)
    pset.addPrimitive(lambda x, y: ts_cov(x, y, n), 2, name='ts_cov_%s' % n)
    pset.addPrimitive(lambda x, y: ts_beta(x, y, n), 2, name='ts_beta_%s' % n)
    pset.addPrimitive(lambda x, y: ts_emal(x, y, n), 2, name='ts_emal_%s' % n)

# 确定优化目标
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 定义一些常量
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# toolbox.register("map", futures.map)


# 分位数函数
def rolling_window_up(a, window=60):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    roll_arr = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    quantile = np.apply_along_axis(lambda x: np.quantile(x, 0.8), 1, roll_arr)
    return np.hstack([[a.max()] * (window - 1), quantile])


def rolling_window_down(a, window=60):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    roll_arr = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    quantile = np.apply_along_axis(lambda x: np.quantile(x, 0.2), 1, roll_arr)
    return np.hstack([[a.min()] * (window - 1), quantile])


def get_max_drawdown(cum_ser):
    Roll_Max = cum_ser.cummax()
    Daily_Drawdown = cum_ser / Roll_Max - 1.0
    return -Daily_Drawdown.min()


def get_calmar_ratio(cum_ser):
    max_draw = get_max_drawdown(cum_ser)
    return (cum_ser.iloc[-1] - 1) / (max_draw + 0.0001)


# 定义评价函数
def win_hold_func(signal, y):
    bools = signal.index[signal != signal.shift(1)]
    win_counts, hold_days = [], []
    for i, j in zip(bools[:-1], bools[1:]):
        sigs = signal[i:j]
        label = sigs.iloc[0]
        if label != 0:
            ret = (y.loc[sigs.index] * label + 1).prod() - 1
            if ret > 0:
                win_counts.append(1)
            else:
                win_counts.append(0)

            hold_days.append(label * len(sigs))
    return np.array(win_counts), np.array(hold_days)


def evalSymbReg_ser(individual, points, y):
    func = toolbox.compile(expr=individual)
    pred_ser = func(*points)
    signal_up = (pred_ser > pred_ser.rolling(60).quantile(0.8)).astype(int)
    signal_down = (pred_ser < pred_ser.rolling(60).quantile(0.2)).astype(int)
    signal = signal_down - signal_up
    cumret = (y * signal + 1).cumprod()
    calma = get_calmar_ratio(cumret)
    win_counts, hold_days = win_hold_func(signal, y)

    hold_day = (hold_days[hold_days > 0]).mean()
    win_rate = win_counts.mean()
    year_count = len(win_counts) / 6

    hold_day = hold_day if not np.isnan(hold_day) else 0
    win_rate = win_rate if not np.isnan(win_rate) else 0

    # print(calma, hold_day, win_rate, year_count)

    # 辅助条件
    condition = 1
    if calma < 0.3 or year_count < 2 or win_rate < 0.5 or hold_day < 6:
        condition = 0

    return calma * np.sqrt(hold_day) * np.sqrt(year_count) * win_rate * condition,  # return tuple


toolbox.register("evaluate", evalSymbReg_ser,
                 points=[df_train_x[col] for col in df_x.columns], y=df_train_y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))


def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    # mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof


# %%
if __name__ == '__main__':
    import pickle
    np.random.seed(2021)
    pop, log, hof = main()
    cp = (pop, hof, toolbox)
    pickle.dump(pop, open("model_pop.pkl", "wb"))
