import unittest
from prepare_data.devide_instances import devide

import numpy as np
import torch

class TestInstsDevide(unittest.TestCase):
    def check_equal_tens(self, t1, t2):
        diff = t1.float() - t2.float()
        self.assertEqual(torch.sum(diff**2), 0)

    def test_small(self):
        example = torch.FloatTensor(np.array([[[1, 2, 3], 
                                              [1, 2, 3],
                                              [3, 2, 1]]]))
    
        matr = example[0]
        
        devided = devide(example)
        self.check_equal_tens(devided[0], example==1)
        self.check_equal_tens(devided[1], example==2)
        self.check_equal_tens(devided[2], example==3)

    def test_medium(self):
        example = torch.FloatTensor(np.array([[[1, 2, 3], 
                                              [4, 5, 6],
                                              [7, 8, 9]]]))
    
        matr = example[0]
        devided = devide(example)
        for i in range(9):
            self.check_equal_tens(devided[i], example==i+1)

