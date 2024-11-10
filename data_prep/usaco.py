import os
import json

os.system("gdown https://drive.google.com/uc?id=1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi")

os.system("unzip data.zip")

with open("./datasets/usaco_subset307_dict.json") as f:
    data = json.load(f)
