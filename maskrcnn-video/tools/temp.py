import numpy as np
from sklearn.metrics import f1_score
import torch


# gt = np.array([[1,0,0,0],[0,0,0,1]])
# pd = np.array([[1,1,1,0],[1,1,1,1]])

gt = np.array([1,0,0,0])
pd = np.array([0,1,1,0])

f1 = f1_score(gt, pd, average='binary')
print(f1)

output_buffer = [[] for i in range(5)]
output_buffer[0].append(1)
# output_buffer = []*5

pd = torch.tensor([1,1,2,3]).numpy()
print(pd)
print('.')