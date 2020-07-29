import torch
import torch.nn as nn

###############
# ELM
###############
class ELM():
    def __init__(self, input_size, h_size, num_classes, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = num_classes
        self._device = device

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = torch.zeros(self._h_size, device=self._device)

        self._activation = torch.sigmoid

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def fit(self, x, t):
        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        #print(H_pinv.shape)
        self._beta = H_pinv.mm(t)
        #self._beta = torch.mul(t,H_pinv)


    def evaluate(self, x, t):
        y_pred = self.predict(x)
        #print(y_pred)
        #print(t)
        #acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        acc = torch.sum(torch.argmax(y_pred, dim = -1)== torch.argmax(t), dim = -1).item() / len(t)
        #acc = torch.sum((t-y_pred)**2)/len(t)
        loss = 0.5 * torch.mean((t - y_pred)**2)
        return acc,loss
    


