
from consts import LAYER_SIZES
import numpy as np
import random

class Nnet:
    def __init__(self):
        self.Ws = ['_']
        self.bs = ['_']
        for it in range(1, len(LAYER_SIZES)):
            mat = []
            for i in range(LAYER_SIZES[it]):
                row = []
                for j in range(LAYER_SIZES[it-1]):
                    row.append(random.uniform(-1, 1))
                mat.append(row)
            self.Ws.append(np.array(mat))
            
            self.bs.append(np.zeros((LAYER_SIZES[it], 1)))