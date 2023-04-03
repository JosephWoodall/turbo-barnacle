import torch.optim as optim

def get_optimizer(model, lr=0.001):
    """

    :param model: 
    :param lr:  (Default value = 0.001)

    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer
