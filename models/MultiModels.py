import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModels(nn.Module):
    def __init__(self, model_list):
        super(MultiModels, self).__init__()
        self.models_0 = model_list[0]
        self.models_1 = model_list[1]
        self.models_2 = model_list[2]

    def __getitem__(self, index):
        if index == 0:
            return self.models_0
        elif index == 1:
            return self.models_1
        elif index == 2:
            return self.models_2