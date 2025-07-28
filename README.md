<br>

<picture><img src="https://github.com/uclamii/model_tuner/blob/main/assets/modeltunersmaller.png?raw=true" width="250" style="border: none; outline: none; box-shadow: none;" oncontextmenu="return false;"></picture>

<br>

[![Downloads](https://pepy.tech/badge/model_tuner)](https://pepy.tech/project/model_tuner) [![PyPI](https://img.shields.io/pypi/v/model_tuner.svg)](https://pypi.org/project/model_tuner/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12727322.svg)](https://doi.org/10.5281/zenodo.12727322)

The `model_tuner` library is a versatile and powerful tool designed to facilitate the training, tuning, and evaluation of machine learning models. It supports various functionalities such as handling imbalanced data, applying different scaling and imputation techniques, calibrating models, and conducting cross-validation. This library is particularly useful for hyperparameter tuning and ensuring optimal performance across different metrics.

## Prerequisites

Before installing `model_tuner`, ensure your system meets the following requirements:

## Python Version

`model_tuner` requires **Python 3.7 or higher**. Specific dependency versions vary depending on your Python version.

## 📦 Dependencies

The following dependencies will be automatically installed for each python version when you install `model_tuner` via pip:

| Package              | Python 3.7               | Python 3.8                | Python 3.9–3.10            | Python 3.11+         |
|----------------------|--------------------------|---------------------------|----------------------------|----------------------|
| `joblib`             | `1.3.2`                  | ✓                         | ✓                          | ✓                    |
| `tqdm`               | `4.66.4`                 | ✓                         | ✓                          | ✓                    |
| `catboost`           | `1.2.7`                  | ✓                         | ✓                          | ✓                    |
| `setuptools`         | —                        | `75.1.0`                  | ✓                          | ✓                    |
| `wheel`              | —                        | `0.44.0`                  | ✓                          | ✓                    |
| `numpy`              | `1.21.4`                 | `>=1.19.5,<2.0.0`         | ✓                          | ✓                    |
| `pandas`             | `1.1.5`                  | `>=1.3.5,<2.2.3`          | ✓                          | ✓                    |
| `scikit-learn`       | `0.23.2`                 | `>=1.0.2,<1.4.0`          | `>=1.0.2,<=1.5.1`          | `1.5.1`              |
| `scipy`              | `1.4.1`                  | `>=1.6.3,<1.11`           | `>=1.6.3,<=1.14.0`         | `1.14.0`             |
| `imbalanced-learn`   | `0.7.0`                  | `0.12.4`                  | ✓                          | ✓                    |
| `scikit-optimize`    | `0.8.1`                  | `0.10.2`                  | ✓                          | ✓                    |
| `xgboost`            | `1.6.2`                  | `2.1.2`                   | ✓                          | ✓                    |

> **Legend:**  
> ✓ – Same as previous version  
> — – Not applicable or not required


## 💾 Installation

You can install `model_tuner` directly from PyPI:

```bash
pip install model_tuner
```

## 📄 Official Documentation

https://uclamii.github.io/model_tuner

## 🌐 Website

https://www.mii.ucla.edu/

## ⚖️ License

`model_tuner` is distributed under the Apache License. See [LICENSE](https://github.com/uclamii/model_tuner?tab=Apache-2.0-1-ov-file) for more information.

## 📚 Citing `model_tuner`

If you use `model_tuner` in your research or projects, please consider citing it.

```bibtex
@software{funnell_2024_12727322,
  author       = {Funnell, Arthur and
                  Shpaner, Leonid and
                  Petousis, Panayiotis},
  title        = {Model Tuner},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.341b},
  doi          = {10.5281/zenodo.12727322},
  url          = {https://doi.org/10.5281/zenodo.12727322}
}
```

## Support

If you have any questions or issues with `model_tuner`, please open an issue on this [GitHub repository](https://github.com/uclamii/model_tuner/).

## Acknowledgements

This work was supported by the UCLA Medical Informatics Institute (MII) and the Clinical and Translational Science Institute (CTSI). Special thanks to Dr. Alex Bui for his invaluable guidance and support, and to Panayiotis Petousis, PhD, for his original contributions to this codebase.
