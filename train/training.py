import torch
from torch import nn

loss_fn = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(params= model_vgg.model.parameters() , lr=0.01) 

torch.cuda.is_available()