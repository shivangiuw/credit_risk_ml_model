
## Credit Risk Analysis Report

Analysis of the performance of machine learning models on original datasets with imbalanced classes vs. on resampled datasets for best predictions.

## Purpose of the analysis:

Build a logistic regression model that can identify the creditworthiness of borrowers using a dataset of historical lending activity and various techniques to train and evaluate models with imbalanced classes as healthy loans easily outnumber risky loans.

## Data:

The classification model will be built upon the dataset of historical lending activity from a peer-to-peer lending services company with financial information for every loan as mentioned below wherein loan status is the classification column(target):

1. loan_size
2. interest_rate
3. borrower_income
4. debt_to_income
5. num_of_accounts
6. derogatory_marks
7. loan_status

`Loan size, interest rate, borrower income, bebt to income ratio, number of accounts and derogatory marks are the features(feature variable "X") of the model.`

`Loan status is target variable("y"), classified with 0 (healthy loan) and 1 (high-risk loan) labels and will be predicted with the same labels.`

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
The target variable "y"(loan_status) is an imbalanced classes with following value counts:
* `0` (healthy loan) - 75036
* `1` (high-risk loan)- 2500
Hence the class is dominated by "0" which makes it difficult for the model to accurately predict the minority class "1".


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


## Results

### * Machine Learning Model 1:

 * Balanced accuracy score:
    0.9520479254722232
    
 * Confusion Matrix: 
     18663,   102,
        56,   563

 * Description of Model 1 Accuracy, Precision, and Recall scores.

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384

### * Machine Learning Model 2:
  
 * Balanced accuracy score:
     0.9936781215845847
 
 * Confusion Matrix:
       18649,   116,
           4,   615
 * Description of Model 2 Accuracy, Precision, and Recall scores.
  
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384

## Evaluation metrics:

` Balanced accuracy score`: Balanced accuracy is a metric that one can use when evaluating how good a binary classifier is. It is especially useful when the classes are imbalanced, i.e. one of the two classes appears a lot more often than the other.


`Confusion Matrix`


`accuracy = (TPs + TNs) ÷ (TPs + TNs + FPs + FNs)`
(The accuracy measures how often the model was correct.)

`precision = TPs ÷ (TPs + FPs)`
(The precision, also known as the positive predictive value (PPV), measures how confident we are that the model correctly made the positive predictions.)

`recall = TPs / (TPs + FNs)`
(The recall measures the number of actually fraudulent transactions that the model correctly classified as fraudulent.)

`F1 = 2 × (precision × recall) ÷ (precision + recall)`
(F1 Score is the weighted average of Precision and Recall.)


## Summary

In the mentioned scenario, it is more important to correctly classify high risk loans(1s) than healthy loans to avoid them. Also to `avoid high risk loans`, it is essential to avoid False negatives for `1s`. Hence, we should focus on the metrics considering `false negatives` more and choose the model which brings down the false negatives.

                    pre       rec       spe        f1       geo       iba       sup
   (model 1)  0    1.00      0.99      0.91      1.00      0.95      0.91     18765
   
   (model 2)  0    1.00      0.99      0.99      1.00      0.99      0.99     18765

For healthy loans (0), all the metrics have improved with model trained with resampled data.

                     pre       rec       spe        f1       geo       iba    sup
  (model 1)  1       0.85      0.91      0.99      0.88      0.95      0.90    619
  
  (model 2)  1       0.84      0.99      0.99      0.91      0.99      0.99    619
 
* The `balanced accuracy` has improved radically  from 0.95 to 0.99 after training the Logistic regression model on resampled   dataset and predicting the target class.

* Looking at the `confusion matrix` the False negatives(`FNs`) dropped from 56 to 4 in the second model, thus the chances of wrongly classifying the High Risk loans as Healthy loans have dropped drastically in the model with resampled dataset.

* Further considering `recall = TPs / (TPs + FNs)`, it improved from 0.91 to 0.99 with the model trained on resampled data whereas `f1` also improved from 0.88 to 0.91.

* Although `precision` dropped a little from 0.85 to 0.84 which seems tolerable with improvements in all other scores.

In view of the above, model 2 i.e. logistic regression model trained with resampled data performs better than model 1 trained on data with imbalanced class, as it not only predicts healthy loans better but also works better at avoiding at wrongly classifying the high risk loans as healthy loans. 




