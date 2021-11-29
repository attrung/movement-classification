import csv
import random
from shutil import copy
import os

train = []
valid = []
with open("data/main_data.csv") as file:
    reader = csv.reader(file, delimiter=',')
    next(reader, None)
    for row in reader:
        if random.random() < 0.1:
            valid.append((row[1], row[2]))
        else:
            train.append((row[1], row[2]))

os.makedirs("train", exist_ok=True)
os.makedirs("valid", exist_ok=True)

print("Copying {} files for train".format(len(train)))
for i in range(len(train)):
    os.makedirs("train/{}".format(train[i][1]), exist_ok=True)
    copy("data/{}".format(train[i][0]), "train/{}/{}.csv".format(train[i][1], i))

print("Copying {} files for valid".format(len(valid)))
for i in range(len(valid)):
    os.makedirs("valid/{}".format(valid[i][1]), exist_ok=True)
    copy("data/{}".format(valid[i][0]), "valid/{}/{}.csv".format(valid[i][1], i))
