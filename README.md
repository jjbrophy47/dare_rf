DaRE RF: Data Removal-Enabled Random Forests
---

**dare** is a python library that implements *machine unlearning* for random forests, enabling the _efficient_ removal of training data without having to retrain from scratch. It is built using Cython and is designed to be scalable to large datasets.

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

# training data
X_train = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]])
y_train = np.array([1, 1, 1, 0, 1])

X_test = np.array([[1, 0]])  # test instance

# train a DaRE RF model
rf = dare.Forest(n_estimators=100,
                 max_depth=3,
                 k=5,  # no. thresholds to consider per attribute
                 topd=0,  # no. random node layers
                 random_state=1)
rf.fit(X_train, y_train)

rf.predict_proba(X_test)  # prediction before deletion => [0.5, 0.5]
rf.delete(3)  # delete training example at index 3 ([1, 0], 0)
rf.predict_proba(X_test)  # prediction after deletion => [0.0, 1.0]
```

Reference
---
<<<<<<< HEAD
Brophy and Lowd. [Machine Unlearning for Random Forests](https://arxiv.org/abs/2009.05567). ICML 2021.
=======
For further details please refer to our ICML 2021 paper: [Machine Unlearning for Random Forests](http://proceedings.mlr.press/v139/brophy21a.html).
>>>>>>> 503421b3bdfd68c326c64ff2973dc61bab024eb0

```
@inproceedings{brophy2021machine,
  title={Machine Unlearning for Random Forests},
  author={Brophy, Jonathan and Lowd, Daniel},
  booktitle={International Conference on Machine Learning},
  pages={1092--1104},
  year={2021},
  organization={PMLR}
}
```
