import unittest
import torch
import numpy as np
from network import TabNetDecoder

class TestDecoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.input_dim = np.random.randint(16,256)
        self.n_d = np.random.randint(16,256)
        self.n_steps = np.random.randint(1,10)
        self.n_independent = np.random.randint(16,256)
        self.n_shared = np.random.randint(1,10)
        self.vbs = np.random.randint(16,256)
        self.test_model = TabNetDecoder(
            self.input_dim,
            self.n_d,
            self.n_steps,
            self.n_independent,
            self.n_shared, 
            self.vbs,
        )
    
    def test_input_dim(self):
        pass
    
    def test_predictive_layer_dimension(self):
        pass

    def test_n_shared(self):
        pass

    def test_vbs(self):
        pass