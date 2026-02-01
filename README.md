# OpenFE: Automated Feature Generation

## Overview

Efficient automated feature generation tool for tabular data.

## Features

- Discovers effective features for GBDT and neural networks
- Supports parallel computing
- 23 useful operators for feature generation
- Supports binary/multi-classification and regression
- Automatically handles missing values and categorical features

## Installation

```bash
pip install openfe
```

**Note:** Do not use `conda install openfe` (installs different package).

## Quick Example

```python
from openfe import OpenFE, transform

ofe = OpenFE()
features = ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)
train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
```

## Documentation

- [Paper](https://arxiv.org/abs/2211.12507)
- [Documentation](https://openfe-document.readthedocs.io/en/latest/)
- [Examples](https://github.com/IIIS-Li-Group/OpenFE/tree/master/examples)

## Results

- Outperforms existing feature generation methods
- IEEE-CIS Fraud Detection: XGBoost with OpenFE features beats 99.3% of teams
