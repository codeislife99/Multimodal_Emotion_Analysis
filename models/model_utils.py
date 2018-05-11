import torch
from .Module import Module
from .utils import clear

# Identity layer - taken from torch.legacy.nn.Identity
class Identity(Module):

    def updateOutput(self, input):
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput
        return self.gradInput

    def clearState(self):
        clear(self, 'gradInput')