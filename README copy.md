DaRE Forests
---

**DaRE** (**Da**ta **R**emoval-**E**nabled) forest (a.k.a. DaRE RF) is a a random forest variant that provides efficient _removal_ of training data without having to retrain from scratch.

Install
---
1. Install Python 3.7+.
1. Install dependencies and compile project.
	* Run `make all`.

Preprocess Data
---
1. Follow the `data/[dataset]/readme.md` to download and preprocess the desired dataset.

Experiments
---
Use the following steps to replicate experimental results.

1. Test the deletion efficiency of a DaRE RF model.
	* Run `python3 scripts/experiments/delete.py` with arguments:
		* `--dataset`: Dataset to use, `surgical`, `vaccine`, `adult`, `bank_marketing`, `flight_delays`, `diabetes`, `no_show`, `olympics`, `census`, `credit_card`, `ctr`, `twitter`, `synthetic`, or `higgs`.
		* `--n_estimators`: No. trees in the forest.
		* `--max_depth`: Maximum depth to build each tree to.
		* `--topd`: No. layers at the top of each tree to use for random nodes.
		* `--k`: No. thresholds to consider per feature.
		* `--subsample_size`: No. samples to consider per deletion.
	* Results are saved to `output/delete/`.

1. Postprocess resuls.
	* Run `python3 scripts/postprocess/delete.py`.
	* Results are saved to `output/delete/csv/`.

1. Visualize results.
	* Run `python3 scripts/plot/plot_cbg.py` with arguments:
		* `--dataset`: List of datasets (with results) to visualize.
		* Results are saved to `output/plots/delete_cbg/`.
