# House Prices - Advanced Regression Techniques

## Submission 1
- Model: K-fold cross validation, k=5
- Parameters:
  - Learning rate: .02
  - Batch size: 16
  - Num epochs: 16
- Results
  - Training data validation error: .0487
  - Test data validation error: .20436
  - Kaggle submission rank: 3470/3961 (bottom 12.4%)

### Analysis
Given the large gap between the training and test data validation errors, we conclude that our model overfit.

To improve our results, we will try to reduce overfitting by reducing the learning rate and number of epochs.

We are curious what effect batch size has on the results, specifically 16 vs 32.


## Submission 2
- Model: K-fold cross validation, k=5
- Parameters:
  - Learning rate: .01
  - Batch size: 16
  - Num epochs: 12
- Results
  - Training data validation error: 0.07637
    - Per fold validation error: `[0.0554, 0.1069, 0.0790, 0.0581, 0.0823]`
  - Test data validation error: .26994

### Analysis
The main question is how overfitting seems to have increased despite reducing the learning rate and number of epochs.

Out of curiosity, we will increase the batch size to 32 to see what that does.


## Submission 3
- Model: K-fold cross validation, k=5
- Parameters:
  - Learning rate: .01
  - Batch size: 32
  - Num epochs: 12
- Results
  - Training data validation error: 0.1113
    - Per fold validation error: `[0.0909, 0.1415, 0.1199, 0.0773, 0.1266]`
  - Test data validation error: .32387

### Analysis
Results seem to suggest that batch size has a similar effect on test vs training data validation error.

We will stick with 16 for now, but if we try 32 again, we should try increasing epochs/lr as well.


## Submission 4
So far we haven't seen any instance where, given two models, one has higher training accuracy than the other but
lower test accuracy. Thus, for now we will assume that lower training data validation error corresponds to lower
test data validation error.

Now, we want to study the data a bit more, first just to see if anything sticks out.

Also, we notice that when k=5, the second fold's validation error seems quite different from the others.
We want to determine why this is. 1460 partitioned into 5 parts is `[[0, 291], [292, 583], ...]`, so the indices in
question are rows 292 through 583. We note the following about those rows:
- Largest lot areas
  - Contains the two largest lot areas (215_245 (314) and 164_660 (336)).
  - Next three largest are (159_000, 115_149, 70_761).
  - Largest lot area in test data is 56_600.

Let's try to remove the 4 largest lot area examples from the data.

Removed 4 largest lot area rows from the data. Reduced k from 5 to 4 so it partitioned evenly.

- Model: K-fold cross validation, k=4
- Parameters:
  - Learning rate: .02
  - Batch size: 16
  - Num epochs: 16
- Results
  - Training data validation error: 0.0516
    - Per fold validation error: `[0.0567, 0.0535, 0.0386, 0.0577]`
  - Test data validation error: .20950


## Submission 5
Let's try pure linear regression.

- Model: Lin regression
- Parameters:
  - Learning rate: .02
  - Batch size: 16
  - Num epochs: 16
- Results
  - Training data validation error: 0.0534
  - Test data validation error: .20901

  
## Submission 6
- Model: Lin regression
- Parameters:
  - Learning rate: .02
  - Batch size: 16
  - Num epochs: 26
- Results
  - Training data validation error: 0.043
  - Test data validation error: .17720
  - Kaggle submission rank: 3356/3961 (bottom 15.3%)

  
## Submission 7
- Model: Lin regression
- Parameters:
  - Learning rate: .02
  - Batch size: 16
  - Num epochs: 32
- Results
  - Training data validation error: 0.0389
  - Test data validation error: .18913


## Submission 8
Let's try an MLP.

- Parameters
  - Learning rate: .02
  - Batch size: 16
  - Num epochs 14
  - Num hidden units 256
- Results
  - Training data validation error: .0129
  - Test data validation error: .32706

### Analysis
This is the first example where the training validation is lower than other submissions but the test validation error
is higher.


## Submission 9


- Parameters
  - Learning rate: .02
  - Batch size: 16
  - Num epochs 32
  - Num hidden units 256
- Results
  - Training data validation error: .0099
  - Test data validation error: .31323


## Submission 9


- Parameters
  - Learning rate: .02
  - Batch size: 16
  - Num epochs 32
  - Num hidden units 256
- Results
  - Training data validation error: .0099
  - Test data validation error: .31323



## Submission 10


- Parameters (not positive here)
  - Learning rate: .02
  - Batch size: 16
  - Num epochs 32
  - Num hidden units 256
  - Dropout .5
- Results
  - Training data validation error: .007
  - Test data validation error: .26742


## Submission 11

K-fold MLP
mlp-k-fold-submission-4.csv
- Parameters
  - Learning rate: .02
  - Batch size: 16
  - Num epochs 64
  - Num hidden units 256
  - Dropout .5
- Results
  - Training data validation error: .0061
  - Test data validation error: .26603


## Data notes
26 Neighborhoods
- Nice:
  - NridgHt
  - NoRidge
- Numerical values with some NAs
  - LotFrontage
  - GarageYrBlt
  - MasVnrArea
