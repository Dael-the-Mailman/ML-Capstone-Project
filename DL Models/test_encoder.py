import unittest
import torch
import numpy as np
from network import TabNetEncoder

class TestEncoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEncoder, self).__init__(*args, **kwargs)
        self.input_dim = np.random.randint(16,256)
        self.output_dim = np.random.randint(16,256)
        self.n_d = np.random.randint(16,256)
        self.n_a = np.random.randint(16,256)
        self.n_steps = np.random.randint(1,10)
        self.gamma = np.random.lognormal(1.3, 1.0)
        self.n_independent = np.random.randint(1,10)
        self.n_shared = np.random.randint(1,10)
        self.vbs = np.random.randint(16,256)

        self.test_model = TabNetEncoder(
            self.input_dim,
            self.output_dim,
            self.n_d,
            self.n_a,
            self.n_steps,
            self.gamma,
            self.n_independent,
            self.n_shared,
            1e-15,
            self.vbs)
    
    def test_input_dim(self):
        self.assertEqual(self.test_model.input_dim, self.input_dim)

    def test_output_dim(self):
        self.assertEqual(self.test_model.output_dim, self.output_dim)

    def test_n_d(self):
        self.assertEqual(self.test_model.n_d, self.n_d)

    def test_n_a(self):
        self.assertEqual(self.test_model.n_a, self.n_a)

    def test_n_steps(self):
        self.assertEqual(self.test_model.n_steps , self.n_steps)
        
    def test_gamma(self):
        self.assertEqual(self.test_model.gamma, self.gamma)
        
    def test_n_independent(self):
        self.assertEqual(self.test_model.n_independent, self.n_independent)

    def test_n_shared(self):
        self.assertEqual(self.test_model.n_shared, self.n_shared)

    def test_vbs(self):
        self.assertEqual(self.test_model.vbs, self.vbs)