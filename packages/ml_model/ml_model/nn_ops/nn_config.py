from torch.nn import BCEWithLogitsLoss
import torch.optim as optim


EPOCHS = 15
LOSS_FUNCTION = BCEWithLogitsLoss()
BATCH_SIZE = 16
LR = 0.01
OPTIMIZER = optim.Adam

SCHEDULER = optim.lr_scheduler.CyclicLR
SCHEDULER_BASE_LR = 0.001
SCHEDULER_MAX_LR = 0.01

NN_NAME = 'nn_model.pt'
USER_DEFINED_CHANGE = 'weightdecay_neurons_512_cyclicLR'