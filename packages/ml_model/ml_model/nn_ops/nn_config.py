from torch.nn import BCEWithLogitsLoss
import torch.optim as optim


EPOCHS = 10
LOSS_FUNCTION = BCEWithLogitsLoss()
BATCH_SIZE = 50
LR = 0.01
OPTIMIZER = optim.Adam
NN_NAME = 'nn_model.pt'