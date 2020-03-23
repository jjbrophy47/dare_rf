"""
Tess the CeDAR tree implementation.
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import cedar
from experiments.utility import data_util, exact_adv_util

seed = 1
n_remove = 100

X_train, X_test, y_train, y_test = data_util.get_data('mfc18_mfc19', seed, data_dir='data')
np.random.seed(seed)
delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)

delete_indices = exact_adv_util.exact_adversary(X_train, y_train, n_samples=n_remove, seed=seed,
                                                verbose=1)

# delete_indices = sorted(delete_indices)
# print(delete_indices)
# delete_indices = [307, 2019, 663, 7593]
# delete_indices = [663, 7593]

# print(delete_indices)
# print(X_train[delete_indices, 43])
# print(X_train[delete_indices, 245])
# print(X_train[delete_indices, 119])
# print(X_train[delete_indices, 178])
# print(X_train[delete_indices, 194])
# print(X_train[delete_indices, 242])
# print(X_train[delete_indices, 302])
# print(X_train[delete_indices, 271])
# print(X_train[delete_indices, 56])
# print(X_train[delete_indices, 55])
# print(X_train[delete_indices, 226])
# print(X_train[delete_indices, 103])
# print(X_train[delete_indices, 141])
# print(X_train[delete_indices, 98])

n_samples = 10
n_features = 2

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
# print(data, np.arange(len(data)))
print(data.shape)

print('data assembled')

delete_indices = np.random.choice(X_train.shape[0], size=n_remove, replace=False)
# print(delete_indices)
print('deleting {} instances'.format(n_remove))

t1 = time.time()
model = cedar.Tree(epsilon=0.5, lmbda=10, max_depth=20, random_state=1).fit(X_train, y_train)
print('build time: {:.7f}s'.format(time.time() - t1))

# model.print()

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
proba = model.predict_proba(X_test)[:, 1]
print('auc: {:.3f}'.format(roc_auc_score(y_test, proba)))

# sequential delete
for i in range(len(delete_indices)):
    # print('\ndeleting {}: {}'.format(delete_indices[i], data[delete_indices[i]]))
    # print('deleting {}'.format(delete_indices[i]))
    t1 = time.time()
    model.delete(delete_indices[i])
    print('delete time: {:.7f}s'.format(time.time() - t1))
    # model.print()

# batch delete
t1 = time.time()
model.delete([delete_indices])
print('\ndelete time: {:.7f}s'.format(time.time() - t1))
print(model.get_removal_statistics())
# model.print()

proba = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('auc: {:.3f}'.format(roc_auc_score(y_test, proba)))
print()
