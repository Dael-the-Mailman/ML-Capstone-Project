import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
import pandas as pd
from network import TabNet
from dotenv import dotenv_values
from tqdm import tqdm

if __name__ == '__main__':

    # Setup and Configuration
    config = dotenv_values("../.env")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 1024
    NUM_EPOCHS = 2

    # Load Data
    labels = pd.read_csv(config["TRAIN_LABELS_PATH"],chunksize=BATCH_SIZE)
    df = pd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part", 
                     chunksize=BATCH_SIZE)

    # PyTorch Setup
    model = TabNet(
            input_dim=2319,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=3,
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

    model.train()
    for epoch in range(1, NUM_EPOCHS):
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
        