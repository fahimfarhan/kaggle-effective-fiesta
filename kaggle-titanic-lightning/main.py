# This is a sample Python script.
import os.path
from typing import Any

import lightning
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
competition_name = "kaggle-titanic-lightning"
local_root = f"/home/soumic/Codes/kaggle/{competition_name}"
kaggle_root = "/kaggle"


def is_local() -> bool:
  return os.path.isdir(local_root)


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

class TitanicDataSets(Dataset):
  def __init__(self, X, y):
    self.X: np.ndarray = X
    self.y = y
    self.mlength = len(y)
    pass

  def __len__(self):
    return self.mlength

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]



class LitBinaryClassifier(LightningModule):
  def __init__(self, model: TitanicSurvivalClassifier, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model
    self.loss = nn.BCEWithLogitsLoss()
    pass

  def training_step(self, batch, batch_nb, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    print(f"\n\n{batch = }")
    x_ara, y_ara = batch
    x = x_ara[0]
    y = y_ara[0]
    print(f"fafafa { x = }")
    print(f"fafafa { y = }")
    loss_value = self.loss(x, y)
    tensorboard_logs = {'train_loss': loss_value}
    return {'loss': loss_value, 'log': tensorboard_logs}


  def forward(self, x, *args: Any, **kwargs: Any) -> Any:
    return self.model(x)

  def configure_optimizers(self) -> OptimizerLRScheduler:
    return torch.optim.Adam(self.parameters(), lr=0.02)

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
  if competition_name == "some_competition_name":
    raise Exception("Update the competition name")

  root = kaggle_root
  if is_local():
    root = local_root
  input_directory = f"{root}/input"
  output_directory = f"{root}/working"

  df = pd.read_csv(f"{input_directory}/train.csv")
  headers = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
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
  y = y1 # np.expand_dims(y1, axis=1)

  print(f"{ type(X) = },\n{ X = }")
  print(y)

  # return

  # val_x = val["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
  # val_y = val["Survived"]

  # return
  titanic_survival_model = TitanicSurvivalClassifier(input_size=len(X1.columns)).double()

  ds = TitanicDataSets(X, y)
  dl = DataLoader(ds)



  litbinclassifier = LitBinaryClassifier(model=titanic_survival_model)
  trainer = lightning.Trainer()
  trainer.fit(model=litbinclassifier, train_dataloaders=dl)
  pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
