# Analysis of the performance of machine learning models on original datasets with imbalanced classes vs. on resampled datasets for best predictions.

Build a logistic regression model that can identify the creditworthiness of borrowers using a dataset of historical lending activity and various techniques to train and evaluate models with imbalanced classes as healthy loans easily outnumber risky loans.

The analysis and the classification will be done first using logistics regression model on the original dataset with imbalanced target class and then on the resampled dataset using RandomOverSampler module form imbalanced-learn. 

## Technologies and Modules

This tool leverages python 3.7 with the following packages:

* [pandas] (https://pandas.pydata.org/docs/getting_started/index.html)- for data analysis
* [scikit-learn] (https://scikit-learn.org/stable/)- open source machine learning library
* [pathlib] (https://docs.python.org/3/library/pathlib.html#module-pathlib)- to read file path
* [imbalanced-learn] (https://imbalanced-learn.org/)-for tools when dealing with classification with imbalanced classes.


## Installation Guide

```
conda install pandas
conda install -c conda-forge imbalanced-learn

```

## Contributor

Shivangi Gupta

## License

MIT
