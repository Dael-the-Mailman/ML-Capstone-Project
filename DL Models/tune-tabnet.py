import torch
import optuna
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import pandas as pd
from network import TabNet
from dotenv import dotenv_values
from datetime import datetime

# Setup and Configuration
config = dotenv_values("../.env")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device, "device")
BATCH_SIZE = 1024
PATIENCE = 3 # How many epochs will we wait until performance gets better or not?
SAVE_PATH = "./model_parameters/"
TIMEOUT = 1*60*60

# Metric
def amex_metric_mod(y_true, y_pred):
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def objective(trial):
    # Model Hyperparameters
    param = {
        "input_dim": 2319,
        "output_dim": 1,
        "n_d": trial.suggest_int("n_d", 4, 64),
        "n_a": trial.suggest_int("n_a", 4, 64),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0, log=True),
        "cat_idxs": [],
        "cat_dims": [],
        "cat_emb_dim": 1,
        "n_independent": 2,
        "n_shared": 2,
        "epsilon": 1e-15,
        "vbs": 128,
        "momentum": trial.suggest_float("momentum", 0.02, 1.0, log=True)
    }
    model = TabNet(**param).to(device)
    optimizer = Adam(model.parameters(), lr=2e-2)

    first_pass = True
    oof_tensors = {}
    best_metric = 0.0 # Keeps track of best metric performance
    patience_count = 0
    for epoch in range(1,101): # Runs maximum of 100 epochs
        # Load Data
        labels = pd.read_csv(config["TRAIN_LABELS_PATH"], 
                             chunksize=BATCH_SIZE)
        df = pd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part", 
                         chunksize=BATCH_SIZE)
        total_loss = 0.0
        model.train()
        for i, (chunk, chunk_labels) in enumerate(zip(df, labels)):
            random = np.random.randint(5) # Determines which entries are going to be used in oof prediction
            x = torch.Tensor(chunk.values).to(device, non_blocking=True)
            y = torch.Tensor(chunk_labels["target"].values).reshape(-1, 1).to(device, non_blocking=True)
            if random == 0 and first_pass:
                # If it's the first pass create the validation set
                oof_tensors[i] = (x, y)
                continue
            if not first_pass and i in oof_tensors.keys():
                # If not the first pass then skip the training on current entry
                continue

            # Train Model
            y_hat, M_loss = model(x)
            loss = F.mse_loss(y_hat, y) - (1e-3*M_loss)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1) # Clip gradient
            optimizer.step()
            total_loss += loss.cpu().detach().numpy().item()
            
            if i % 2 == 0:
                optimizer.zero_grad(set_to_none=True)
        # Validate 
        model.eval()
        preds = []
        labels = []
        for x, y in list(oof_tensors.values()):
            y_hat, _ = model(x)
            preds += y_hat.cpu().detach().numpy().flatten().tolist()
            labels += y.cpu().detach().numpy().flatten().tolist()
        metric = amex_metric_mod(labels, preds)
        print(f"Epoch {epoch} | train_loss: {total_loss / (i-len(oof_tensors)+1):.4f} | validation_metric: {metric:.4f}")
        first_pass = False

        # Saves model based on performance over time
        if metric > best_metric:
            best_metric = metric
            patience_count = 0
            id = datetime.now().strftime("%d-%m-%Y-%H%M%S")
            torch.save(model.state_dict(), SAVE_PATH+f"Optuna_TabNet_{best_metric:.4f}_{id}.pt")
        else:
            patience_count += 1
        
        # If model hasn't improved in given time training stops
        if patience_count >= PATIENCE:
            print("Early Stopping Activated!!!")
            break
    
    return best_metric


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=TIMEOUT)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # model = TabNet(
    #         input_dim=2319,
    #         output_dim=1,
    #         n_d=4,
    #         n_a=4,
    #         n_steps=3,
    #         gamma=1.3,
    #         cat_idxs=[],
    #         cat_dims=[],
    #         cat_emb_dim=1,
    #         n_independent=2,
    #         n_shared=2,
    #         epsilon=1e-15,
    #         vbs=128,
    #         momentum=0.02,
    #     ).to(device)
    # optimizer = Adam(model.parameters(), lr=2e-2)

    # first_pass = True
    # oof_tensors = {}
    # best_metric = 0.0 # Keeps track of best metric performance
    # patience = 3
    # patience_count = 0
    # for epoch in range(1,101):
    #     # Load Data
    #     labels = pd.read_csv(config["TRAIN_LABELS_PATH"], chunksize=1024)
    #     df = pd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part", chunksize=1024)
    #     total_loss = 0.0
    #     model.train()
    #     for i, (chunk, chunk_labels) in enumerate(zip(df, labels)):
    #         random = np.random.randint(5) # Determines which entries are going to be used in oof prediction
    #         x = torch.Tensor(chunk.values).to(device, non_blocking=True)
    #         y = torch.Tensor(chunk_labels["target"].values).reshape(-1, 1).to(device, non_blocking=True)
    #         if random == 0 and first_pass:
    #             # If it's the first pass create the validation set
    #             oof_tensors[i] = (x, y)
    #             continue
    #         if not first_pass and i in oof_tensors.keys():
    #             # If not the first pass then skip the training on current entry
    #             continue

    #         # Train Model
    #         y_hat, M_loss = model(x)
    #         loss = F.mse_loss(y_hat, y) - (1e-3*M_loss)
    #         loss.backward()
    #         clip_grad_norm_(model.parameters(), 1) # Clip gradient
    #         optimizer.step()
    #         total_loss += loss.cpu().detach().numpy().item()
            
    #         if i % 2 == 0:
    #             optimizer.zero_grad(set_to_none=True)
    #     # Validate 
    #     model.eval()
    #     preds = []
    #     labels = []
    #     for x, y in list(oof_tensors.values()):
    #         y_hat, _ = model(x)
    #         preds += y_hat.cpu().detach().numpy().flatten().tolist()
    #         labels += y.cpu().detach().numpy().flatten().tolist()
    #     metric = amex_metric_mod(labels, preds)
    #     print(f"Epoch {epoch} | train_loss: {total_loss / (i-len(oof_tensors)+1):.4f} | validation_metric: {metric:.4f}")
    #     first_pass = False

    #     if metric > best_metric:
    #         best_metric = metric
    #         patience_count = 0
    #     else:
    #         patience_count += 1
        
    #     if patience_count >= patience:
    #         print("Early Stopping Activated!!!")
    #         break
    # print(f"Best Performance Score: {best_metric:.4f}")


    
