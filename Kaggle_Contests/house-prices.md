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



## Data notes
26 Neighborhoods
- Nice:
  - NridgHt
  - NoRidge
- Numerical values with some NAs
  - LotFrontage
  - GarageYrBlt
  - MasVnrArea
