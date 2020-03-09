import time
import numpy as np

from splitter import Splitter
from tree import Tree
from tree import DepthFirstTreeBuilder


# def split_np(X, y):

#     count = 0
#     pos_count = 0

#     l_total, lp_total = 0, 0
#     r_total, rp_total = 0, 0

#     count = len(y)
#     pos_count = np.sum(y)

#     # compute statistics for each attribute
#     for j in range(n_features):

#         left_count = 0
#         left_pos_count = 0

#         left_indices = np.where(X[:, j] == 1)[0]
#         left_count = len(left_indices)
#         left_pos_count = y[left_indices].sum()

#         right_count = count - left_count
#         right_pos_count = pos_count - left_pos_count

#         l_total += left_count
#         lp_total += left_pos_count
#         r_total += right_count
#         rp_total += right_pos_count

#     print('left total: {}, right total: {}'.format(left_count, right_count))
#     print('left pos total: {}, right pos total: {}'.format(left_pos_count, right_pos_count))


n_samples = 10000000
n_features = 20

# X = np.random.randint(2, size=(n_samples, n_features), dtype=np.int32)
# y = np.random.randint(2, size=n_samples, dtype=np.int32)

np.random.seed(1)
X = np.random.randint(2, size=(n_samples, n_features), dtype=np.int32)
np.random.seed(1)
y = np.random.randint(2, size=n_samples, dtype=np.int32)

data = np.hstack([X, y.reshape(-1, 1)])
print(data)

X = np.asfortranarray(X, dtype=np.int32)
y = np.ascontiguousarray(y, dtype=np.int32)
f = np.ascontiguousarray(np.arange(X.shape[1]), dtype=np.int32)

print('data assembled')

# n_features = X.shape[1]
# X = np.asfortranarray(X, dtype=np.int32)
# y = np.ascontiguousarray(y, dtype=np.int32)

t1 = time.time()
tree = Tree(2)
splitter = Splitter(2, 10, 1)
builder = DepthFirstTreeBuilder(splitter, 2, 1, 10)
builder.build(tree, X, y, f)
print('time: {:.7f}s'.format(time.time() - t1))

# print(tree.values)
# print(tree.n_samples)
# print(tree.features)
# print(tree.left_children)
# print(tree.right_children)
print(tree.max_depth)
