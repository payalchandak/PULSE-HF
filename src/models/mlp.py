from torch import nn

class MLP(nn.Module):
    def __init__(self, layers, dropout_prob):
        super().__init__()
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1])) 
            modules.append(nn.ReLU()) 
            if i < len(layers) - 2:
                modules.append(nn.Dropout(dropout_prob))
        modules.pop(-1) # we don't want an activation after the final layer
        self.model = nn.Sequential(*modules) 

    def forward(self, x):
        return self.model(x)
