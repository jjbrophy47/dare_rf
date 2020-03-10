import time
import numpy as np

from splitter import _Splitter
from tree import _Tree
from tree import _TreeBuilder


n_samples = 10
n_features = 2

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

X_train = np.asfortranarray(X_train, dtype=np.int32)
y_train = np.ascontiguousarray(y_train, dtype=np.int32)
f = np.ascontiguousarray(np.arange(X_train.shape[1]), dtype=np.int32)

print('data assembled')

t1 = time.time()
tree = _Tree(2)
splitter = _Splitter(1, 10, 1)
builder = _TreeBuilder(splitter, 2, 1, -1)
builder.build(tree, X_train, y_train, f)
print('time: {:.7f}s'.format(time.time() - t1))

print('nodes: {}'.format(tree.n_nodes))
print('leaf values: {}'.format(tree.values))
print('counts: {}'.format(tree.counts))
print('pos counts: {}'.format(tree.pos_counts))

for i in range(tree.n_nodes):
    if tree.chosen_features[i] != -2:
        print('node {}:'.format(i))
        print('  left counts: {}'.format(tree._get_left_counts(i)))
        print('  left pos counts: {}'.format(tree._get_left_pos_counts(i)))
        print('  right counts: {}'.format(tree._get_right_counts(i)))
        print('  right pos counts: {}'.format(tree._get_right_pos_counts(i)))
        print('  features: {}'.format(tree._get_features(i)))

print('feature counts: {}'.format(tree.feature_counts))
print('chosen features: {}'.format(tree.chosen_features))
print('max depth: {}'.format(tree.max_depth))

preds = tree.predict(X_test)
print(preds)
print(y_test)
