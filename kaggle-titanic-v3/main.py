from typing import Any

import torch
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch import optim
import math
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

timber = logging.getLogger("titanic_logger")
timber.setLevel(logging.INFO)


class TitanicSurvivalClassifier(nn.Module):
    def __init__(self, input_size=7, num_classes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sequential = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        # print(f"forward -> { x = }")
        return self.sequential(x)


class TitanicLightningModule(LightningModule):
    def __init__(self, input_size=7, num_classes=1, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.sequential = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=num_classes)
        )

        self.loss_function = nn.BCEWithLogitsLoss()
        pass

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        self.log("my_metric", x.mean())
        logits = self(x)
        loss = self.loss_function(logits, y)
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard = self.logger.experiment
        # tensorboard.any_summary_writer_method_you_want()  # todo: Fix it
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, *args: Any, **kwargs: Any) -> Any:
        return self.sequential(x)


def get_xy(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    X1 = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    # X1 = df[["Sex", "Age"]]

    # print(f"{len(X1.columns) = }")
    # return
    X = X1.values
    y1 = df["Survived"].values * 1.0
    y = np.expand_dims(y1, axis=1)
    return X, y

class TitanicDataSet(Dataset):
  def __init__(self, x: np.ndarray, y: np.ndarray):
    self.x = x.astype(np.float32)
    self.y = y.astype(np.float32)
    pass

  def __len__(self):
    return self.y.size

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

class TitanicDataModule(LightningDataModule):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.batch_size = 16
        train, validate = train_test_split(df, test_size=0.2)
        train_x, train_y = get_xy(df=train)
        val_x, val_y = get_xy(df=validate)

        self.train_loader = DataLoader(TitanicDataSet(train_x, train_y), batch_size=self.batch_size, shuffle=True,
                                       num_workers=7, persistent_workers=True)
        self.validate_loader = DataLoader(TitanicDataSet(val_x, val_y), batch_size=self.batch_size, shuffle=True,
                                          num_workers=7, persistent_workers=True)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        timber.info(f"inside setup: {stage = }")
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.validate_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return None


# from sklearn import
# import skorch
def cabin_ohe(cabin: str) -> int:
    try:
        if cabin == float("nan"):
            print("cabin == nan, returning")
            return 0
        number = int(cabin[1:])
        character = (1 + ord(cabin[0]) - ord('A'))
        output = int(character * 1000 + number)
        return output
    except Exception as x:
        mynan = float("nan")
        print(f"{ cabin = }, { type(cabin) = }, { mynan = },{ type(mynan) = }, { x = }")
        return 0


def cabins_ohe(cabins: pd.Series) -> list:
    output: list = [cabin_ohe(cabin) for cabin in cabins]
    return output


def replace_nan_with_avg(input: pd.Series) -> pd.Series:
    output = input.fillna(input.mean())
    return output


def generic_ohe(input: pd.Series) -> list:
    mset = input.unique()
    dict = {}
    i = 0
    for item in mset:
        dict[item] = i

    mlist: list = [dict[item] for item in input]
    return mlist


def start():
    df = pd.read_csv("input/train.csv")
    headers = [
        "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
        "Embarked"
    ]

    # df["Cabin"] = cabins_ohe(df["Cabin"])  # too complex, ignore for now
    # print(df["Cabin"])

    generic_ohe_needed_headers = [
        "Sex", "Embarked"
    ]

    replace_nan_with_avg_headers = [
        "Age", "SibSp", "Parch", "Fare"
    ]

    for column in replace_nan_with_avg_headers:
        df[column] = replace_nan_with_avg(df[column])

    drop_columns = [
        "Name", "Ticket", "Cabin"
    ]

    df = df.drop(drop_columns, axis=1)
    # for column_name in drop_columns:

    for column_name in generic_ohe_needed_headers:
        df[column_name] = generic_ohe(df[column_name])

    print(df.head())

    # train, val = train_test_split(df, train_size=0.8)

    X1 = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    # X1 = df[["Sex", "Age"]]

    # print(f"{len(X1.columns) = }")
    # return
    X = X1.values
    y1 = df["Survived"].values * 1.0
    y = np.expand_dims(y1, axis=1)

    # print(f"{ type(X) = },\n{ X = }")
    # print(y)

    # return

    # val_x = val["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # val_y = val["Survived"]

    # titanic_survival_model = TitanicSurvivalClassifier(input_size=len(X1.columns)).double()

    titanic_data_module = TitanicDataModule(df=df)
    net = TitanicLightningModule(input_size=len(X1.columns))
    net = net.float()
    net = net.to("mps")
    trainer = Trainer(log_every_n_steps=5) #// (gpus=1)
    trainer.fit(model=net, datamodule=titanic_data_module)
    pass


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    start()
    # print(cabin_ohe("C454"))
    pass
