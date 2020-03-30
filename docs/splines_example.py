import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parma import polyharmonic_interpolator, polyharmonic_hermite_interpolator


def f(x):
    return np.sin(x)

def df(x):
    return np.cos(x)

data_locs = -2/3 + 2*np.arange(3)/3
data_vals = f(data_locs)
data_hermite_vals = df(data_locs)

x = np.linspace(-1, 1, 20)
hermite_splines, _ = polyharmonic_hermite_interpolator(
    (data_locs,), data_vals, (0,), (data_hermite_vals,), 5)
lagrange_splines = polyharmonic_interpolator((data_locs,), data_vals, 2)

fig, ax = plt.subplots()
pd.Series(f(x), x, name='sin').plot(ax=ax)
pd.DataFrame([lagrange_splines((x,)), hermite_splines((x,))],
             ['lagrange', 'hermite'], x).T.plot(ax=ax, linestyle='--')
pd.Series(data_vals, data_locs, name='samples'
          ).plot(ax=ax, marker='o', color='k', linewidth=0)
ax.legend()
