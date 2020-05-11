import numpy as np
from math import log2


def diff(a, b, n):
    pp = a / n
    pn = b / n
    d1 = (- pp * safe_log2(pp)) + (- pn * safe_log2(pn))

    pp = (a - 1) / (n - 1)
    pn = b / (n - 1)
    d2 = (- pp * safe_log2(pp)) + (- pn * safe_log2(pn))

    return np.abs(d1 - d2)


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return log2(x)


def upper_bound(n):
    return (2 / n) * safe_log2(n)


if __name__ == '__main__':
    n = 3

    ub = upper_bound(n)
    print('upper bound={:5}'.format(ub))

    a = 1
    b = n - 1
    case1 = diff(a=a, b=b, n=n)
    print('n={}, a={:5}, b={:5}, diff={}'.format(n, a, b, case1))

    a = n / 2
    b = n / 2
    case2 = diff(a=a, b=b, n=n)
    print('n={}, a={:5}, b={:5}, diff={}'.format(n, a, b, case2))
