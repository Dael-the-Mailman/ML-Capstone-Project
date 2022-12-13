import unittest
import numpy as np
import torch
from network import GBN

class TestGBN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGBN,self).__init__(*args, **kwargs)
        self.input_dim = np.random.randint(16,256)
        self.vbs = np.random.randint(32,256)
        self.momentum = np.random.rand()
        
        self.test_model = GBN(
            self.input_dim,
            self.vbs,
            self.momentum
        )

    def test_input_dim(self):
        self.assertEqual(self.test_model.input_dim, self.input_dim)

    def test_vbs(self):
        self.assertEqual(self.test_model.vbs, self.vbs)

    def test_momentum(self):
        self.assertEqual(self.test_model.momentum, self.momentum)

    def test_forward(self):
        x = torch.rand(np.random.randint(2048, 8192), self.input_dim)
        y = self.test_model(x)
        self.assertEqual(x.shape, y.shape)