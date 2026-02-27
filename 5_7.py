import pandas as pd
import torch
from torch import nn
import lib.d2l as d2l

"""
Kaggle House Price Prediction

The training dataset includes 1460 examples, 80 features, and one label,
while the validation data contains 1459 examples and 80 features.

"""

class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))

    def preprocess(self):
        """
        Pandas Ref
            Series: One-dimensional ndarray with axis labels (including time series).
            DataFrame: Two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).

        """
        # Remove the ID and label columns.
        label = "SalePrice"
        features = pd.concat(  # Returns a DataFrame.
            (self.raw_train.drop(columns=["Id", label]),
             self.raw_val.drop(columns=["Id"])))

        # Standardize numerical columns.
        #
        # print(features)
        # print(features.dtypes)
        # MSSubClass      int64
        # MSZoning        str
        # LotFrontage     float64
        # LotArea         int64
        # Street          str
        """
        Columns with mixed types are stored with the object dtype.
        """
        # numeric_features = features.dtypes[exclude["object"]].index
        numeric_dtypes = features.select_dtypes(exclude=["object"])
        numeric_features = numeric_dtypes.columns.tolist()
        # numeric_features = features.columns.get_indexer(numeric_column_names)

        print(numeric_features)
        # print(numeric_features.index)
        # print(numeric_features)
        # print(features.dtypes)
        # return

        features[numeric_features] = features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))

        # Replace NAN numerical features by 0 (thereby setting them to the feature mean).
        features[numeric_features] = features[numeric_features].fillna(0)

        # Replace discrete features by one-hot encoding.
        features = pd.get_dummies(features, dummy_na=True)

        # Save preprocessed features.
        self.train = features[:self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]
        self.val = features[self.raw_train.shape[0]:].copy()
        return features.copy()

    def get_dataloader(self, train):
        label = "SalePrice"
        data = self.train if train else self.val
        if label not in data: return
        get_tensor = lambda x: torch.tensor(x.values.astype(float), dtype=torch.float32)
        # Logarithm of prices
        tensors = (
            get_tensor(data.drop(columns=[label])),  # X
            torch.log(get_tensor(data[label])).reshape((-1, 1))  # Y
        )
        return self.get_tensorloader(tensors, train)



def k_fold_data(data, k):
    """
    Returns the ith fold of the data in a K-fold cross-validation procedure.
    It slices out the ith segment as validation data and returns the rest as training data.
    """

    # KaggleHouse data modules.
    rets = []

    # The (max) partition size of the rows in X when split into k groups.
    fold_size = data.train.shape[0] // k  # // is the floor division operator, i.e. divide, then floor.

    # For each group of rows:
    for j in range(k):
        # The indices of the rows in the jth group.
        idx = range(j * fold_size, (j + 1) * fold_size)
        rets.append(
            KaggleHouse(
                data.batch_size,                # batch size
                data.train.drop(index=idx),     # train, i.e. all rows except the jth group
                data.train.loc[idx]             # val, i.e. the jth group
            )
        )

    return rets


def k_fold(trainer, data, k, lr):
    """
    Average validation error is returned when we train K times in the K-fold cross-validation.
    """

    val_loss = []
    models = []

    # For the k KaggleHouse data modules returned by k_fold_data:
    for i, data_fold in enumerate(k_fold_data(data, k)):
        # Create a linear regression model.
        model = d2l.LinearRegression(lr)
        model.board.yscale = "log"

        # Unclear why we don't want to display any board beyond the first.
        if i != 0: model.board.display = False

        # Train the model using the kth KaggleHouse data module.
        trainer.fit(model, data_fold)

        # Add the final validation loss value from the model to our val_loss array.
        val_loss.append(float(model.board.data["val_loss"][-1].y))

        # Add the model to our models array.
        models.append(model)

    print(val_loss)
    print(f"Average validation log MSE = {sum(val_loss) / len(val_loss)}")

    return models
    # return sum(val_loss) / len(val_loss)



def compute_avg_loss(params):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()
    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    val_loss = k_fold(trainer, data, k=params["k"], lr=params["lr"])
    return val_loss


# learning_rates = [0.005, 0.01, 0.02, 0.03]  # , 0.06, 0.08, 0.1]
# losses = []
# for lr in learning_rates:
#     hyper_parameters = {
#         "batch_size": 32,
#         "lr": lr,
#         "k": 5,
#         "num_epochs": 20
#     }
#     hyper_parameters["loss"] = compute_avg_loss(hyper_parameters)
#     losses.append(hyper_parameters)



# print(losses)
params = {
    "batch_size": 32,
    "lr": 0.01,
    "k": 5,
    "num_epochs": 12
}
data = KaggleHouse(batch_size=params["batch_size"])
preprocessed_data = data.preprocess()
# preprocessed_data.to_csv('preprocessed_data.csv', index=False)
#
# trainer = d2l.Trainer(max_epochs=params["num_epochs"])
# # val_loss = k_fold(trainer, data, k=params["k"], lr=params["lr"])
# # return val_loss
# models = k_fold(trainer, data, k=params["k"], lr=params["lr"])
# # return models
#
# preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))
#          for model in models]
# # Taking exponentiation of predictions in the logarithm scale
# ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)
# submission = pd.DataFrame({'Id':data.raw_val.Id,
#                            'SalePrice':ensemble_preds.detach().numpy()})
# submission.to_csv('submission-3.csv', index=False)

"""
Hyperparameter Analysis

    - A lr between 0.005 - 0.03 seems optimal.
        - 0.02 seems to perform the best under most conditions.
        - Higher seems to cause over/underflow issues.
        - 0.005 always seems worse than higher values, so we assume going even lower won't help.
    
    - Batch sizes of 16 or 32 seems optimal.
        - 8 and 64 perform worse when tested on a variety of lr and batch size combinations.
        - 16 performs a bit better than 32 at around 10 epochs, but 32 starts to catch up around 16 epochs.
        - 32 starts to perform worse at 18 epochs, so 32 and 16 epochs seems best.
        - 16 seems to perform best at 12-18 epochs with lr=0.02.
        - Lower learning rates seem to tolerate higher num_epochs better.
    
Solid choice:
    - Lr = 0.02
    - Batch size = 16
    - Num epochs = 16
    - Result: .2
    

==================

K fold losses with first sub values: 
[0.03522426262497902, 0.076180599629879, 0.04366099834442139, 0.033396344631910324, 0.04994451254606247]
[0.05072057992219925, 0.07779184728860855, 0.04780907928943634, 0.03473088517785072, 0.05592658743262291]

k=10



"batch_size": 32,
"lr": 0.01,
"k": 10,
"num_epochs": 16
[0.08303093910217285, 0.06910906732082367, 0.11860129982233047, 0.11547736823558807, 0.08537093549966812,
 0.07593133300542831, 0.08127707242965698, 0.06016569212079048, 0.12053171545267105, 0.10609477758407593]

"""

"""
Numeric features:
['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
"""
