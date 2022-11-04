import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from torch.optim import Adam
from network import TabNet
from dotenv import dotenv_values
from utils import AMEXDataset

class TabNetRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        vbs=128,
        momentum=0.02
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.vbs = vbs
        self.momentum = momentum
        self.learning_rate = 0.02;

        self.tabnet = TabNet(
            input_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            [],
            [],
            cat_emb_dim,
            n_independent,
            n_shared,
            epsilon,
            vbs,
            momentum,
        )

    def forward(self, x):
        out, _ =self.tabnet(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    # model = TabNetRegressor(

    # )
    config = dotenv_values("../.env")
    chunk_size = 16384
    
    df = pd.read_csv(config["WRANGLED_DATA"] + "scaled_train/train-0.csv.part", 
                     chunksize=chunk_size)

    for i, chunk in enumerate(df):
        print(chunk.iloc[0])