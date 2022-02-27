# Analysis of the performance of machine learning models on original datasets with imbalanced classes vs. on resampled datasets for best predictions.

The purpose of the analysis is to build a logistic regression model that can identify the creditworthiness of borrowers using a dataset of historical lending activity and various techniques to train and evaluate models with imbalanced classes as healthy loans easily outnumber risky loans.

The analysis and the classification will be done first using logistics regression model on the original dataset with imbalanced target class and then on the resampled dataset using RandomOverSampler module form imbalanced-learn. 

## Steps followed in the machine learning process:

### 1. Split the Data
- Split the features variable data `X` and Target variable `y` into Training `(X_train, y_train)` and Testing Sets`(X_test, y_test)` using `test_train_split` function.

### 2. Create a Logistic Regression Model with the Original Data
- Instantiate the `Logistic Regression model` 
- Fit a logistic regression model by using the training data train the model on X_train and y_train `(imbalanced)` datasets wherein y_train has following value counts: `0`- 56271, `1`- 1881 .
- Save the predictions ("y") on the testing data labels by using the testing feature data (X_test) and the fitted model.
- Evaluate the `model’s performance` by following:
1. accuracy score of the model 
2. confusion matrix 
3. classification report

### 3. Predict a Logistic Regression Model with Resampled Training Data
- Use the `RandomOverSampler` module from the `imbalanced-learn library` to resample the data.
1. Instantiate the random oversampler model
2. Fit the original training data to the random_oversampler model to get new variable datasets as `X_resampled, y_resampled` from train datasets `X_train, y_train`.
3. Confirm that value counts of the labels have an equal number of data points now i.e. `0`- 56271, `1 `- 56271.

- Use the LogisticRegression classifier and the resampled data(X_resampled, y_resampled) to fit the model and make predictions on X_test.
- Evaluate the model’s performance by doing the following:
1. accuracy score of the model 
2. confusion matrix 
3. classification report
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
conda install -c intel scikit-learn

```

## Contributor

Shivangi Gupta

## License

MIT
