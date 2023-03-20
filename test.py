import torch
x = torch.rand(5, 3)
print(x)

print("=============>")
print("CUDA enabled {0}".format(torch.cuda.is_available()))