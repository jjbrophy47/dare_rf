import os
import sys

import numpy as np
from sklearn.datasets import load_iris

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import dart


def main():

    # get simple dataset
    data = load_iris()
    X = data['data']
    y = data['target']

    # make into binary classification dataset
    indices = np.where(y != 2)[0]
    X = X[indices]
    y = y[indices]

    print(X)

    # train decision tree
    model = dart.Tree(topd=0, k=25, max_depth=2, random_state=1)
    model = model.fit(X, y)

    # predict
    print(X[0])
    print(model.predict_proba(X[[0]]))


if __name__ == '__main__':
    main()
