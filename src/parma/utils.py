from itertools import chain


def polynomial_powers(degree, n_dims):
    powers = ((i,) for i in range(degree + 1))
    for _ in range(n_dims - 1):
        powers = chain(*([(*i, j) for j in range(degree + 1 - sum(i))]
                         for i in powers))
    return [] if n_dims <= 0 else list(powers)
