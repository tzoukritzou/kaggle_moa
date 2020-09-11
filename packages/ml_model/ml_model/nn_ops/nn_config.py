from torch.nn import BCEWithLogitsLoss


EPOCHS = 10
LOSS_FUNCTION = BCEWithLogitsLoss()
BATCH_SIZE = 10
NN_NAME = 'nn_model.pt'