import pandas as pd
import torch
from torch import nn
import lib.d2l as d2l
import xgboost as xgb

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

        # Drop rows with Ids: 250, 314, 336, ane 707
        # self.raw_train.drop([249, 313, 335, 706], inplace=True)
        # self.raw_train.reset_index(inplace=True)

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


class MLP(d2l.Module):
    def __init__(self, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.LazyLinear(1))

    def loss(self, Y_hat, Y, averaged=True):
        loss = nn.MSELoss()
        return loss(Y_hat, Y) if not averaged else loss(Y_hat, Y) / len(Y)

    # def configure_optimizers(self):
    #     param_groups = [
    #         {'params': [p for name, p in self.named_parameters() if 'weight' in name], 'weight_decay': .01},
    #         {'params': [p for name, p in self.named_parameters() if 'bias' in name or 'bn' in name],
    #          'weight_decay': 0.0}
    #     ]
    #     return torch.optim.SGD(param_groups, lr=self.lr)


class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)


def k_fold_data(data, k):
    """
    Returns the ith fold of the data in a K-fold cross-validation procedure.
    It slices out the ith segment as validation data and returns the rest as training data.
    """

    # KaggleHouse data modules.
    rets = []

    # The (max) partition size of the rows in X when split into k groups.
    fold_size = data.train.shape[0] // k  # // is the floor division operator, i.e. divide, then floor.
    print(f"Fold size: {fold_size}")

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


def k_fold(trainer, data, params):
    """
    Average validation error is returned when we train K times in the K-fold cross-validation.
    """

    val_loss = []
    models = []

    # For the k KaggleHouse data modules returned by k_fold_data:
    for i, data_fold in enumerate(k_fold_data(data, params["k"])):
        # Create a linear regression model.
        # model = d2l.LinearRegression(lr)
        # model = WeightDecay(wd=3, lr=params["lr"])
        model = MLP(params["num_hiddens"], params["lr"])
        model.board.yscale = "log"

        # Unclear why we don't want to display any board beyond the first.
        if i != 0: model.board.display = False

        # Train the model using the kth KaggleHouse data module.
        trainer.fit(model, data_fold)

        # Add the final validation loss value from the model to our val_loss array.
        val_loss.append(float(model.board.data["val_loss"][-1].y))

        # Add the model to our models array.
        models.append(model)

    avg_val_loss = sum(val_loss) / len(val_loss)

    return models, val_loss, avg_val_loss


def compute_avg_loss(params):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()
    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    val_loss = k_fold(trainer, data, k=params["k"], lr=params["lr"])
    return val_loss


def create_submission(params, filename):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()
    # print(f"Data val shape: {data.val.shape}")
    # return
    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    models, val_loss, avg_val_loss = k_fold(trainer, data, params)
    print(f"Val losses: {val_loss}")
    print(f"Avg val loss: {avg_val_loss}")
    preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))
             for model in models]
    # Taking exponentiation of predictions in the logarithm scale
    ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)
    submission = pd.DataFrame({'Id': data.raw_val.Id,
                               'SalePrice': ensemble_preds.detach().numpy()})
    submission.to_csv(filename, index=False)


def test_k_fold_params(params):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()

    print(f"Data train shape: {data.train.shape}")

    trainer = d2l.Trainer(max_epochs=params["num_epochs"])

    models, val_loss, avg_val_loss = k_fold(trainer, data, params)
    print(f"Val loss: {val_loss}")
    print(f"Avg val loss: {avg_val_loss}")

    # models, val_loss, avg_val_loss = k_fold(trainer, data, k=params["k"], lr=params["lr"])
    #
    # print(f"Val losses: {val_loss}")
    # print(f"Avg val loss: {avg_val_loss}")


def plain_lin_regression(trainer, data, lr):
    # model = d2l.LinearRegression(lr)
    model = WeightDecay(wd=3, lr=lr)
    model.board.yscale = "log"
    trainer.fit(model, data)
    val_loss = float(model.board.data["val_loss"][-1].y)
    return model, val_loss


def test_lin_regression_params(params):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()
    idx = range(1200, 1460)
    data = KaggleHouse(
        params["batch_size"],
        data.train.drop(index=idx),
        data.train.loc[idx]
    )
    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    model, val_loss = plain_lin_regression(trainer, data, lr=params["lr"])
    d2l.plt.show()
    print(f"Val loss: {val_loss}")


def create_lin_regression_submission(params, filename):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()
    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    model = WeightDecay(wd=3, lr=params["lr"])
    model.board.yscale = "log"
    trainer.fit(model, data)
    preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))]
    # Taking exponentiation of predictions in the logarithm scale
    ensemble_preds = torch.exp(preds[0])
    submission = pd.DataFrame({'Id': data.raw_val.Id,
                               'SalePrice': ensemble_preds.detach().numpy().flatten()})
    submission.to_csv(filename, index=False)


def test_mlp_params(params):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()
    idx = range(1200, 1460)
    data = KaggleHouse(
        params["batch_size"],
        data.train.drop(index=idx),
        data.train.loc[idx]
    )

    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    model = MLP(params["num_hiddens"], params["lr"])
    model.board.yscale = "log"
    trainer.fit(model, data)
    val_loss = float(model.board.data["val_loss"][-1].y)
    d2l.plt.show()
    print(f"Val loss: {val_loss}")


def create_mlp_submission(params, filename):
    data = KaggleHouse(batch_size=params["batch_size"])
    data.preprocess()

    trainer = d2l.Trainer(max_epochs=params["num_epochs"])
    model = MLP(params["num_hiddens"], params["lr"])
    model.board.yscale = "log"
    trainer.fit(model, data)

    preds = [model(torch.tensor(data.val.values.astype(float), dtype=torch.float32))]
    # Taking exponentiation of predictions in the logarithm scale
    ensemble_preds = torch.exp(preds[0])
    submission = pd.DataFrame({'Id': data.raw_val.Id,
                               'SalePrice': ensemble_preds.detach().numpy().flatten()})
    submission.to_csv(filename, index=False)


def test_xgboost_params(params):
    data = KaggleHouse(batch_size=20)
    data.preprocess()
    idx = range(1200, 1460)
    train_df = data.train.drop(index=idx)
    val_df = data.train.loc[idx]
    train_data = xgb.DMatrix(
        train_df.drop(columns=["SalePrice"]),
        label=train_df["SalePrice"]
    )
    val_data = xgb.DMatrix(
        val_df.drop(columns=["SalePrice"]),
        label=val_df["SalePrice"]
    )
    model = xgb.train(
        params,
        train_data,
        num_boost_round=50000,
        evals=[(val_data, "validation")],
        early_stopping_rounds=5
    )
    predictions = model.predict(xgb.DMatrix(data.val))

    preds = [model.predict(xgb.DMatrix(data.val))]
    # Taking exponentiation of predictions in the logarithm scale
    submission = pd.DataFrame({'Id': data.raw_val.Id,
                               'SalePrice': model.predict(xgb.DMatrix(data.val)).flatten()})
    submission.to_csv("xgboost-3.csv", index=False)
    print(predictions)
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score}")


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
# params = {
#     "batch_size": 20,
#     "lr": 0.02,
#     "k": 5,
#     "num_epochs": 26,
#     "num_hiddens": 256
# }
params = {
    "booster": "gblinear",
    "device": "cuda",
    "max_depth": 6,
    "eta": .2,
    "objective": "reg:squaredlogerror",
    "min_child_weight": 0
}

# test_k_fold_params(params)
test_xgboost_params(params)


"""
K-fold Lin w/Weight Decay
16, 26, 0.02, wd=3
.02789

Lin w/Weight Decay
16, 26, 0.02, wd=2
.0371 | .0429

Lin w/Weight Decay
16, 26, 0.02, wd=3
.0367 | .0368 | .037

Lin w/Weight Decay
16, 28, 0.02, wd=3
.0440

Lin w/Weight Decay
16, 24, 0.02, wd=3
.0422

Lin w/Weight Decay
16, 26, 0.02, wd=4
.0394 | .0433

16, 16, 0.02
val loss: 0.0534

16, 26, .02
val loss: .043

16, 30, .02
val loss: .041

16, 32, .02
val loss: 0.0389

16, 26, .015
val loss: .0508

16, 26, .02
val loss: .0506

32, 18, 0.01
val loss: .1008

32, 18, 0.02
val loss: .065

32, 20, 0.02
val loss: .0617

32, 22, 0.02
val loss: .0578

32, 26, 0.02
val loss: .0570

32, 20, 0.025
val loss: .089

32, 26, .015
val loss: .0717

MLP

16, 10, 0.02, 256
val loss: .0165

16, 12, 0.02, 256
val loss: .0131

16, 14, 0.02, 256
val loss: .0129

K-Fold MLP
16, 14, 0.02, 256, k=5
val loss: .0103

K-Fold MLP
16, 14, 0.02, 128, k=5
val loss: .01000

K-Fold MLP
16, 16, 0.02, 128, k=5
val loss: .0093

K-Fold MLP
16, 20, 0.02, 128, k=5
val loss: .0083

K-Fold MLP
16, 32, 0.02, 128, k=5
val loss: .00656

K-Fold MLP
16, 32, 0.02, 128, k=5, dropoout=0.5
val loss: .0092

K-Fold MLP
16, 32, 0.02, 256, k=5, dropoout=0.5
val loss: .0077 | .0095

MLP
16, 32, 0.02, 256, k=5, dropoout=0.5
val loss: .013

MLP
16, 40, 0.02, 256, k=5, dropoout=0.5
val loss: .0092

MLP
16, 64, 0.02, 256, k=5, dropoout=0.5
val loss: .0055

K-Fold MLP
16, 64, 0.02, 256, k=5, dropoout=0.5
val loss: .00797

K-Fold MLP
32, 64, 0.02, 256, k=5, dropoout=0.5
val loss: .011



"""


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
