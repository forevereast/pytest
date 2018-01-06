# coding: utf-8

from __future__  import print_function
import torch

x=torch.Tensor(5,3)
print(x,x.size())

y= torch.rand(5,3)
print (y,y.size())
#
print(x+y)
print(torch.add(x,y))

result = torch.Tensor(4,3)
torch.add(x,y,out=result)
# print(result)

# adds x to y
y.add_(x)
print(y)
print(x[0,:])