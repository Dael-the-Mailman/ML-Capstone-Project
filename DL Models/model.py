import torch
from network import TabNet

MODEL_PATH = "./model_parameters/Optuna_TabNet_0.7822_05-11-2022-181211.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
param = {
        "input_dim": 2319,
        "output_dim": 1,
        "n_d": 13,
        "n_a": 9,
        "n_steps": 9,
        "gamma": 1.3077154313342185,
        "cat_idxs": [],
        "cat_dims": [],
        "cat_emb_dim": 1,
        "n_independent": 2,
        "n_shared": 2,
        "epsilon": 1e-15,
        "vbs": 128,
        "momentum": 0.03072144877400083
    }

print("LOAD MODEL")
model = TabNet(**param).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(torch.device("cpu"))
model.eval()

import pandas as pd
from dotenv import dotenv_values
config = dotenv_values("../.env")
df = pd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part", 
                         chunksize=1024)
for chunk in df:
    x = torch.Tensor([chunk.values[0]])
    print(model(x))
    break