import deep_learning as dl
import torch

#Example
print('Testing pad function')
X = torch.ones(3, 2, 4, 4)
pad_res = dl.pad(X, 2)
print(X)
print(pad_res)
print()

print('Testing cross_correlation function')

#Example 
print('\nExample 1')
X = torch.ones((1, 7, 7), dtype = torch.float64)
K = torch.randint(1, 5, (1, 2, 2), dtype = torch.float64)
print(X)
print(K)
print(dl.cross_correlation(X, K, 2))

# Example from book D2L page 253
print('Example 2')
X = torch.tensor([
    [
     [0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]
    ],
    [
     [1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]
    ]
])

K = torch.tensor([
    [
     [0, 1],
     [2, 3],
    ],
    [
     [1, 2],
     [3, 4]
    ]
])

print(dl.cross_correlation(X, K, 1))
print()


