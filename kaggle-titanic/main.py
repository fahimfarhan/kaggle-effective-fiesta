import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from torch import nn
from torch import optim
import math


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


def get_callbacks() -> list:
  # metric.auc ( uses trapezoidal rule) gave an error: x is neither increasing, nor decreasing. so I had to remove it
  return [
    ("tr_acc", EpochScoring(
      metrics.accuracy_score,
      lower_is_better=False,
      on_train=True,
      name="train_acc",
    )),

    ("tr_recall", EpochScoring(
      metrics.recall_score,
      lower_is_better=False,
      on_train=True,
      name="train_recall",
    )),
    # ("tr_precision", EpochScoring(
    #   metrics.precision_score,
    #   lower_is_better=False,
    #   on_train=True,
    #   name="train_precision",
    # )),
    ("tr_roc_auc", EpochScoring(
      metrics.roc_auc_score,
      lower_is_better=False,
      on_train=False,
      name="tr_auc"
    )),
    ("tr_f1", EpochScoring(
      metrics.f1_score,
      lower_is_better=False,
      on_train=False,
      name="tr_f1"
    )),
    # ("valid_acc1", EpochScoring(
    #   metrics.accuracy_score,
    #   lower_is_better=False,
    #   on_train=False,
    #   name="valid_acc1",
    # )),
    ("valid_recall", EpochScoring(
      metrics.recall_score,
      lower_is_better=False,
      on_train=False,
      name="valid_recall",
    )),
    # ("valid_precision", EpochScoring(
    #   metrics.precision_score,
    #   lower_is_better=False,
    #   on_train=False,
    #   name="valid_precision",
    # )),
    ("valid_roc_auc", EpochScoring(
      metrics.roc_auc_score,
      lower_is_better=False,
      on_train=False,
      name="valid_auc"
    )),
    ("valid_f1", EpochScoring(
      metrics.f1_score,
      lower_is_better=False,
      on_train=False,
      name="valid_f1"
    ))
  ]


def test(net: NeuralNetClassifier):
  print("inside test")
  df = pd.read_csv("data/test.csv")
  headers = [
    "PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
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

  # X1 = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
  X1 = df[["Sex", "Age"]]

  # print(f"{len(X1.columns) = }")
  # return
  X = torch.tensor(X1.values)

  passengerIds = df["PassengerId"]

  y = net.predict(X)
  print(f"{ y.shape = }")
  y = y.squeeze(1)
  print(f"{ y.shape = }")

  result = pd.DataFrame()
  result["PassengerId"] = passengerIds
  result["Survived"] = y

  print(f"{ result.head() = }")
  result.to_csv("data/results.csv", index=False)
  pass


def start():
  df = pd.read_csv("data/train.csv")
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
  y = np.expand_dims(y1, axis=1)

  # print(f"{ type(X) = },\n{ X = }")
  # print(y)

  # return

  # val_x = val["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
  # val_y = val["Survived"]

  titanic_survival_model = TitanicSurvivalClassifier(input_size=len(X1.columns)).double()

  net = NeuralNetClassifier(
    titanic_survival_model,
    max_epochs=20,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam,
    # lr=0.01,
    lr=0.005,
    optimizer__weight_decay=1e-5,  # this is the correct way of passing the
    # optimizer__momentum_decay=0.5,  # weight_decay, momentum_decay etc to NAdam optimizer
    batch_size=16,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    # train_split=0.8,
    verbose=True,
    callbacks=get_callbacks()
  )

  net.fit(X, y)

  # test the model

  # titanic_survival_model.eval()
  # predicted = net.predict(X)
  # print(f"{ predicted = }")
  test(net=net)

  pass


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
  start()
  # print(cabin_ohe("C454"))
  pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
