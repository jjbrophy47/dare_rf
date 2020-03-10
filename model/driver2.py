"""
Tess the CeDAR tree implementation.
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score

import cedar

n_samples = 10
n_features = 2

# generate data
np.random.seed(1)
X_train = np.random.randint(2, size=(n_samples, n_features), dtype=np.int32)
np.random.seed(1)
y_train = np.random.randint(2, size=n_samples, dtype=np.int32)

np.random.seed(2)
X_test = np.random.randint(2, size=(10, n_features), dtype=np.int32)
np.random.seed(2)
y_test = np.random.randint(2, size=10, dtype=np.int32)

# data = np.hstack([X_train, y_train.reshape(-1, 1)])
# print(data)

print('data assembled')

t1 = time.time()
model = cedar.Tree(lmbda=10, max_depth=None, random_state=1).fit(X_train, y_train)
print('build time: {:.7f}s'.format(time.time() - t1))

model.print_tree(show_nodes=False, show_metadata=False)

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
