DaRE RF: Data Removal-Enabled Random Forests
---

**dare_rf** is a python library that implements *machine unlearning* for random forests, enabling the _efficient_ removal of training data without having to retrain from scratch. It is built using Cython and is designed to be scalable to large datasets.

<p align="center">
	<img align="center" src="images/thumbnail.png" alt="thumbnail", width="350">
</p>

Installation
---
1. Install Python 3.7+.
1. Install dependencies and compile project. Run `make all`.

Usage
---
Simple example of removing a single training instance:

```python
import dare
import numpy as np

# initialize some training data
X = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]])
y = np.array([1, 1, 1, 0, 1])

# create a test example
X_test = np.array([[1, 0]])

# train a DaRE RF model
rf = dare.Forest(n_estimators=100,
                 max_depth=3,
                 k=5,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
rf.fit(X, y)

# prediction before deletion => [0.5, 0.5]
rf.predict_proba(X_test)

# delete training example at index 3 ([1, 0], 0)
rf.delete(3)

# prediction after deletion => [0.0, 1.0]
rf.predict_proba(X_test)
```

Reference
---
Brophy and Lowd. [Machine Unlearning for Random Forests](https://arxiv.org/abs/2009.05567). ICML 2021.

```
@article{brophy2021darerf,
  title={Machine Unlearning for Random Forests},
  author={Brophy, Jonathan and Lowd, Daniel},
  journal={arXiv preprint arXiv:2009.05567v2},
  year={2021}
}
```