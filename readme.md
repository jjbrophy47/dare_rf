# CeDAR Trees

**Ce**rtified **D**ata **A**ddition and **R**emoval for decision trees.

Getting Started
---
1. Clone or fork this repository.
1. Install dependencies:
    1. Install Python >= 3.7.3.
    1. Install CeDAR and other Python3 modules: `make all`.
1. Look at `experiments/notebooks/example.pynb` for example usage!

---
Changelog
===

### [0.2.0] - 2020-06-18
* Re-implementation of CeDAR with three differen budget
  allocation strategies: *Single*, *Layer*, and *Pyramid*.

### [0.1.1] - 2020-04-28
* Added entropy as a possible splitting criterion.

### [0.1.0] - 2020-04-24
* Removed division of lambda (dividing lambda by 5, no.
  trees, and max depth) from CeDAR.