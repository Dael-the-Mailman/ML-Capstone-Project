import unittest
import torch
import numpy as np
from network import TabNetDecoder

class TestDecoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDecoder, self).__init__(*args, **kwargs)
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
        self.assertEqual(self.test_model.input_dim, self.input_dim)
    
    def test_predictive_layer_dimension(self):
        self.assertEqual(self.test_model.n_d, self.n_d)

    def test_n_steps(self):
        self.assertEqual(len(self.test_model.feat_transformers), self.n_steps)

    def test_n_independent(self):
        self.assertEqual(self.test_model.n_independent, self.n_independent)

    def test_n_shared(self):
        self.assertEqual(self.test_model.n_shared, self.n_shared)

    def test_vbs(self):
        self.assertEqual(self.test_model.vbs, self.vbs)
    