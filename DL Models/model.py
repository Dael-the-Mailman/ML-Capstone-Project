import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import pandas as pd
from network import TabNet
from dotenv import dotenv_values
from datetime import datetime
from tqdm import tqdm

# Setup and Configuration
config = dotenv_values("../.env")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", device, "device")

def train_loop(
    model, 
    optimizer,
    batch_size=1024,
    num_epochs=100,
    data_path=config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part",
    label_path=config["TRAIN_LABELS_PATH"],
    save_path="./model_parameters/"):

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, num_epochs):
        # Load Data
        labels = pd.read_csv(label_path, chunksize=batch_size)
        df = pd.read_csv(data_path, chunksize=batch_size)
        total_loss = 0.0
        for i, (chunk, chunk_labels) in tqdm(enumerate(zip(df, labels))):
            # Convert CSV data to Tensor on GPU
            x = torch.Tensor(chunk.values).to(device, non_blocking=True)
            y = torch.Tensor(chunk_labels["target"].values).reshape(-1, 1).to(device, non_blocking=True)

            # Calculate loss
            y_hat, M_loss = model(x)
            loss = F.mse_loss(y_hat, y) - (1e-3*M_loss)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1) # Clip gradient
            optimizer.step()
            total_loss += loss.cpu().detach().numpy().item()
            
            if i % 2 == 0:
                optimizer.zero_grad(set_to_none=True)
        
        print(f"Epoch {epoch} | loss: {total_loss / (i+1):.4f}")
        del labels, df
    
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    # Hyperparameter
    BATCH_SIZE = 2048
    NUM_EPOCHS = 10
    SAVE_PATH = "./model_parameters/"
    
    # PyTorch Setup
    model = TabNet(
            input_dim=2319,
            output_dim=1,
            n_d=64,
            n_a=64,
            n_steps=10,
            gamma=1.3,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            vbs=128,
            momentum=0.02,
        ).to(device)
    optimizer = Adam(model.parameters(), lr=2e-2)
    id = datetime.now().strftime("%d-%m-%Y-%H%M%S")
        
    train_loop(
        model=model, 
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        save_path=SAVE_PATH + f"MaxTabNet_{id}.pt" 
    )

    # model.train()
    # for epoch in range(1, NUM_EPOCHS):
    #     total_loss = 0.0
    #     for i, (chunk, chunk_labels) in tqdm(enumerate(zip(df, labels))):
    #         # Convert CSV data to Tensor on GPU
    #         x = torch.Tensor(chunk.values).to(device, non_blocking=True)
    #         y = torch.Tensor(chunk_labels["target"].values).reshape(-1, 1).to(device, non_blocking=True)

    #         # Calculate loss
    #         y_hat, M_loss = model(x)
    #         loss = F.mse_loss(y_hat, y) - (1e-3*M_loss)
    #         loss.backward()
    #         clip_grad_norm_(model.parameters(), 1) # Clip gradient
    #         optimizer.step()
    #         total_loss += loss.cpu().detach().numpy().item()
            
    #         if i % 2 == 0:
    #             optimizer.zero_grad(set_to_none=True)
            
    #     print(f"Epoch {epoch} | loss: {total_loss / (i+1):.4f}")
        