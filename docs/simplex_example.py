import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from parma.regularisation import orthonormalise_data
from parma.utils import standard_simplex, right_inverse


z = np.random.randn(1000, 2)
x = z @ np.array([[1, 0], [0.75, 0.66]]).T

mean_vec = np.mean(x, axis=0)
cov_mat = np.cov(x.T)
total_var = np.trace(cov_mat)
orth_mat = orthonormalise_data(cov_mat)
simplex = (np.sqrt(total_var)*standard_simplex(2)
           @ right_inverse(orth_mat).T + mean_vec)

fig, ax = plt.subplots()
pd.DataFrame(*x.T[::-1]).plot(marker='o', alpha=0.25, linewidth=0,
                              legend=False, ax=ax)
pd.Series(*np.append(simplex, simplex[:1], axis=0).T[::-1]).plot(ax=ax, c='k')
pd.Series([0], [0]).plot(ax=ax, c='k', marker='o')
