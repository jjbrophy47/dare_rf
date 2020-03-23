"""
Tess the CeDAR forest implementation.
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import cedar

from experiments.utility import data_util

seed = 1
n_remove = 10000
n_estimators = 100

batch = True

X_train, X_test, y_train, y_test = data_util.get_data('mfc19', seed, data_dir='data')
np.random.seed(seed)

# n_samples = 1000
# n_features = 20

# # generate data
# np.random.seed(1)
# X_train = np.random.randint(2, size=(n_samples, n_features), dtype=np.int32)
# np.random.seed(1)
# y_train = np.random.randint(2, size=n_samples, dtype=np.int32)

# np.random.seed(2)
# X_test = np.random.randint(2, size=(10, n_features), dtype=np.int32)
# np.random.seed(2)
# y_test = np.random.randint(2, size=10, dtype=np.int32)

data = np.hstack([X_train, y_train.reshape(-1, 1)])
print(data.shape)

print('data assembled')


t1 = time.time()
m1 = RandomForestClassifier(n_estimators=n_estimators, max_depth=20, random_state=1).fit(X_train, y_train)
print('[SK] build time: {:.7f}s'.format(time.time() - t1))
preds = m1.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))

t1 = time.time()
model = cedar.Forest(epsilon=0.1, lmbda=100, n_estimators=n_estimators,
                     max_depth=20, max_features='sqrt', random_state=1).fit(X_train, y_train)
print('\nbuild time: {:.7f}s'.format(time.time() - t1))
# model.print(show_nodes=True, show_metadata=False)

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))


delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)

if not batch:

    # sequential delete
    for i in range(len(delete_indices)):
        t1 = time.time()
        model.delete(delete_indices[i])
        print('delete time: {:.7f}s'.format(time.time() - t1))

else:

    # batch delete
    t1 = time.time()
    model.delete([delete_indices])
    print('\ndelete time: {:.7f}s'.format(time.time() - t1))

print(model.get_removal_statistics())

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
