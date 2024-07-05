# This is a sample Python script.
import os.path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
competition_name = "talkingdata-mobile-user-demographics"
local_root = f"/home/soumic/Codes/kaggle/{competition_name}"
kaggle_root = "/kaggle"


def is_local() -> bool:
  return os.path.isdir(local_root)


def start():
  root = kaggle_root
  if is_local():
    root = local_root
  input_directory = f"{root}/input"
  output_directory = f"{root}/working"


  pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
