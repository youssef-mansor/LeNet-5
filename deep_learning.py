from d2l import torch as d2l # for HyperParameters class
import torch
from torch import nn

def pad(X, padding): #optimized to use slicing instead of nested loops for element-wise assignments
    if(len(X.shape) == 4):
        h = X.shape[2]
        w = X.shape[3]
        padded_X = torch.zeros(X.shape[0], X.shape[1], 2*padding + h, 2*padding + w)
        for idx in range(X.shape[0]): #idx of example in batch size #TODO deal with multilpe channels
            for c in range(X.shape[1]):
                padded_X[idx, c, padding:padding+h, padding:padding+w] = X[idx, c] #shallow copy
            
        return padded_X
    else: #2 dimension
        h = X.shape[0]
        w = X.shape[1]
        padded_X = torch.zeros(2*padding + h, 2*padding + w)
        padded_X[padding:padding+h, padding:padding+w] = X #shallow copy
        return padded_X
    

def cross_correlation(X, K, s):# X(channel, height, width)
    assert K.shape[1] == K.shape[2] #Kernel must be square
    assert K.shape[0] == X.shape[0] #Kernel and Feature has same number of input channels
    ci = X.shape[0]
    k = K.shape[1]
    n = X.shape[1]
    assert k <= n #kernel must be less than or equal to size of input
    o = (n - k + s) //s
    output = torch.zeros((o, o), dtype = torch.float64)
    for a in range(0, o):
        for b in range(0, o):
            i = s*a
            j = s*b
            output[a, b] = (X[:, i:i+k, j:j+k] * K).sum()
    return output

class Conv(d2l.HyperParameters, nn.Module):
    
    def __init__(self, output_channels, padding = 2, kernel_size = 5, K_data = None, s = 1): #for kernal 5 padding 2 is suitable
        super().__init__()
        self.save_hyperparameters()
        if K_data is not None:
            assert tuple([K_data.shape[0], K_data.shape[2], K_data.shape[3]]) == (self.output_channels, self.kernel_size, self.kernel_size)
        self.kernel = nn.Parameter(data = K_data, requires_grad=True)
    
    def forward(self, X): #shape (batch, channels, height, width)
        assert X.shape[2] == X.shape[3] #image is square
        padded_X = pad(X, self.padding)
        input_channels = X.shape[1]
        print()
        if self.kernel.data.nelement() == 0:
            self.kernel.data = torch.randn((self.output_channels, input_channels, self.kernel_size, self.kernel_size), dtype = torch.float64)
        batch_size = padded_X.shape[0]
        w = (X.shape[2] -  self.kernel.shape[2] + self.padding*2 + self.s)//self.s
        h = w
        output = torch.zeros((batch_size, self.output_channels, h, w), dtype = torch.float64)
        for idx in range(batch_size):
            for c in range(self.output_channels):
                output[idx][c] = cross_correlation(padded_X[idx], self.kernel[c], self.s)
        return output
    
class Sigmoid(d2l.HyperParameters, nn.Module):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
    def forward(self, X):#of shape (example, channel, height, width)
        output = torch.zeros_like(X)
        for idx in range(X.shape[0]):
            for c in range(X.shape[1]):
                output[idx, c] = torch.sigmoid(X[idx, c])
        return output

class AvgPool2d(d2l.HyperParameters, nn.Module):
    def __init__(self, kernel_size = 2, method = 'avg'):
        super().__init__()
        self.save_hyperparameters()
        self.k = torch.ones((kernel_size, kernel_size), dtype = torch.float64)
        self.k = self.k * 1/(kernel_size**2)
    def forward(self, X): #shape (example, channel, height, width)
        stride = 2
        sub = int(self.k.shape[0]/2)
        output = torch.zeros((X.shape[0], X.shape[1], int(X.shape[2]/stride), int(X.shape[3]/stride)), dtype = torch.float64)
        for idx in range(X.shape[0]):
            for c in range(X.shape[1]):
                output[idx, c] = cross_correlation(X[idx, c].reshape((1, X[idx, c].shape[0], X[idx, c].shape[1])), self.k.reshape((1, self.k.shape[0], self.k.shape[1])), stride)
        return output
    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def  forward(self, X):
        return X.reshape((X.shape[0], -1))
    
class Linear(nn.Module, d2l.HyperParameters):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.save_hyperparameters()
        self.W = nn.Parameter(data = torch.randn((n_output, n_input), dtype = torch.float64), requires_grad=True)
        self.b = nn.Parameter(data = torch.zeros((n_output, 1), dtype = torch.float64), requires_grad=True)
        
    def forward(self, X): #shape (example, data)
        output = torch.zeros(X.shape[0], self.n_output, dtype = torch.float64)
        for idx in range(output.shape[0]):
            output[idx] = (torch.matmul(self.W, X[idx].reshape((len(X[idx]), 1))) + self.b).reshape((self.n_output))
        return output
    
class Sequential(d2l.HyperParameters, nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.save_hyperparameters()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    def forward(self, X):#(example, channel, height, width)
        for module in self.children():
            X = module(X)
            # print(X.shape)
        return X
    
# class Optim_SGD(d2l.HyperParameters):
#     def __init__(self, params, lr):
#         self.save_hyperparameters()
#         print('hey there in optimizer')

#     def step(self):
#         for param in self.params:
#             print('param before updating', param)
#             param -= self.lr * param.grad
#             print('param after updating', param)
#     def zero_grad(self):
#         for param in self.params:
#             if param.grad is not None:
#                 param.zero_grade_()

class LeNet(nn.Module, d2l.HyperParameters):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = Sequential(Conv(output_channels=6, padding = 2, kernel_size=5), Sigmoid(),
                 AvgPool2d(kernel_size=2, method='avg'), #stride = 2
                 Conv(output_channels=16, padding = 0, kernel_size = 5), Sigmoid(),
                 AvgPool2d(),
                 Flatten(),
                 Linear(400, 120), Sigmoid(),
                 Linear(120, 84), Sigmoid(),
                 Linear(84, 10)
                )
        # self.optim = Optim_SGD(self.parameters(), self.lr)
        # self.optim = torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, X):
        return torch.softmax(self.net(X), 1)
    
    def loss(self, Y_hat, Y): #for a single exmaple more efficient since some terms will vanish anyway
        return -torch.log(Y_hat[list(range(len(Y_hat))), Y.to(dtype = int)]).sum()

    # def configure_optimizers(self):
    #     return Optim_SGD(self.parameters(), self.lr)