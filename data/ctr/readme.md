CTR (Click-Through Rate) Dataset
---
This dataset consists of one day of online ad clicks (day 0), [dataset homepage](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

* To download this dataset, run the following command:
	* `wget -c http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_0.gz`, then unzip this file.

* Preprocess the data.
	* Run `python3 continuous.py`.

This creates a `continuous/train.csv` and `continuous/test.csv`.
