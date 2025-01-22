import copy
from math import inf


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = inf
        self.best_model_state = None
        self.early_stop = False
        self.restore_best_weights = restore_best_weights

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            # store model params corresponding to best loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(
                        self.best_model_state
                    )  # revert to best params
