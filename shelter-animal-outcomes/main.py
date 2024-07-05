# This is a sample Python script.
import os.path
from collections import Counter
from datetime import datetime

# import pandas as pd
# from pandas import DataFrame

import polars as pd
from polars import DataFrame
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import torch
from skorch.callbacks import EpochScoring
from torch import nn
from torch import optim
import skorch

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
competition_name = "some_competition_name"
local_root = f"/home/soumic/Codes/kaggle/{competition_name}"
kaggle_root = "/kaggle"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class ShelterAnimalClassifier(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #  6 inputs
    # self.sequential = nn.Sequential(
    self.h1 = nn.Linear(in_features=6, out_features=512)
    self.h2 = nn.Sigmoid()
      # nn.Dropout(0.3),

    self.h3 = nn.Linear(in_features=512, out_features=1024)
    self.h4 = nn.Sigmoid()
      # nn.Dropout(0.3),

    self.h5 = nn.Linear(in_features=1024, out_features=5)
    self.h6 = nn.Sigmoid()
      # nn.Dropout(0.3),

    self.h7 = nn.Linear(in_features=5, out_features=1)
    # )
    # 5 outputs
    pass

  def forward(self, x):
    # y = self.sequential(x)
    x = x.to(torch.float32)
    h = self.h1(x)
    h = self.h2(h)
    h = self.h3(h)
    h = self.h4(h)
    h = self.h5(h)
    h = self.h6(h)
    y = self.h7(h)
    return y


def is_local() -> bool:
  return os.path.isdir(local_root)


def date_time_preprocessing(some_column: pd.Series) -> pd.Series:
  date_format = "%Y-%m-%d %H:%M:%S"
  mlist: list = [ int(datetime.strptime(item, date_format).timestamp()) for item in some_column]
  return pd.Series(mlist)


def age_preprocessing(some_age: str) -> int:
  if some_age is None:
    return 365  # todo: replace with avg I guess
  ara = some_age.split(" ")
  value = int(ara[0])
  unit = ara[1]
  if unit == "years" or unit == "year":
    return value * 365
  elif unit == "months" or unit == "month":
    return value * 30
  elif unit == "weeks" or unit == "week":
    return value * 7
  return 0


def age_preprocessing_column(some_column: pd.Series) -> pd.Series:
  mlist = [age_preprocessing(age) for age in some_column]
  return pd.Series(mlist)


def generic_ohe(some_column: pd.Series) -> pd.Series:
  mset = some_column.unique()
  mdict = {}
  i = 0
  for item in mset:
    mdict[item] = i
    i += 1

  mlist: list = [mdict[item] for item in some_column]
  return pd.Series(mlist)


def start():
  root = kaggle_root
  if is_local():
    root = local_root
  input_directory = f"{root}/input"
  output_directory = f"{root}/working"

  all_headers = [
    "AnimalID", "Name", "DateTime", "OutcomeType", "OutcomeSubtype", "AnimalType",
    "SexuponOutcome", "AgeuponOutcome", "Breed", "Color"
  ]

  X_headers = ["DateTime", "AnimalType",
               "SexuponOutcome", "AgeuponOutcome", "Breed", "Color"]
  y_headers = ["OutcomeType"]
  drop_columns = ["Name", "OutcomeSubtype"]

  input_df: DataFrame = pd.read_csv("input/train.csv.gz")
  for c in drop_columns:
    input_df = input_df.drop(c)

  generic_ohe_columns = ["AnimalType", "SexuponOutcome", "Breed", "Color"]
  for header in generic_ohe_columns:
    # input_df[header] = generic_ohe(input_df[header])
    new_column = generic_ohe(input_df[header])
    input_df = input_df.with_columns(new_column.alias(header))

  print(input_df.head())
  print(input_df["AnimalType"])
  new_column = date_time_preprocessing(input_df["DateTime"])
  input_df = input_df.with_columns(new_column.alias("DateTime"))
  print(input_df["DateTime"])

  new_column = age_preprocessing_column(input_df["AgeuponOutcome"])
  input_df = input_df.with_columns(new_column.alias("AgeuponOutcome"))
  print(input_df["AgeuponOutcome"])

  y = input_df["OutcomeType"]
  y = LabelEncoder().fit_transform(y) * float(1)
  print(type(y))
  print(f"{y.shape = }")

  print(Counter(input_df['OutcomeType']))
  print(Counter(y))

  label_map = {
    'Adoption': 0, 'Transfer': 4, 'Return_to_owner': 3, 'Euthanasia': 2, 'Died': 1  # alphabetic?
  }
  print(input_df.head())
  print(f"{y = }")
  X = input_df[X_headers]
  X = X.to_numpy() * float(1)
  print(X)

  m_criterion = nn.CrossEntropyLoss()
  m_optimizer = optim.Adam

  m_model = ShelterAnimalClassifier() # .float()

  net = skorch.NeuralNetClassifier(
    module=m_model,
    criterion=m_criterion,
    optimizer=m_optimizer,
    lr=0.01,
    # optimizer__weight_decay=1e-5,
    iterator_train__shuffle=True,
    batch_size=16,
    verbose=True,
    # device=DEVICE
  )

  net.fit(X, y)

  pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
