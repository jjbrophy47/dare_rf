"""
Tess the CeDAR forest implementation.
"""
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import cedar

n_samples = 100
n_features = 100
n_estimators = 2

# generate data
np.random.seed(1)
X_train = np.random.randint(2, size=(n_samples, n_features), dtype=np.int32)
np.random.seed(1)
y_train = np.random.randint(2, size=n_samples, dtype=np.int32)

np.random.seed(2)
X_test = np.random.randint(2, size=(10, n_features), dtype=np.int32)
np.random.seed(2)
y_test = np.random.randint(2, size=10, dtype=np.int32)

data = np.hstack([X_train, y_train.reshape(-1, 1)])
print(data)

print('data assembled')

t1 = time.time()
m1 = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=1).fit(X_train, y_train)
print('build time: {:.7f}s'.format(time.time() - t1))
preds = m1.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))

t1 = time.time()
model = cedar.Forest(epsilon=0.01, lmbda=10, n_estimators=n_estimators,
                     max_depth=10, max_features='sqrt', random_state=1).fit(X_train, y_train)
print('\nbuild time: {:.7f}s'.format(time.time() - t1))
# model.print(show_nodes=True, show_metadata=False)

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))

# remove instances
t1 = time.time()
model.delete(0)
print('\ndelete time: {:.7f}s'.format(time.time() - t1))
# model.print(show_nodes=True, show_metadata=False)

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))

# remove instances
t1 = time.time()
model.delete(1)
print('\ndelete time: {:.7f}s'.format(time.time() - t1))
# model.print(show_nodes=True, show_metadata=False)

preds = model.predict(X_test)
print('accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))

types, depths = model.get_removal_statistics()
print(types)
print(depths)
