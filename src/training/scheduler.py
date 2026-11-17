import numpy as np

def make_lr_lambda(batch_size, n_epochs, warmup_ratio = 0.1):
    
    total_steps = batch_size * n_epochs
    warmup_steps = warmup_ratio * total_steps
    
    def lr_lambda_decay(step):
        scale = min(1.0, (step + 1) / warmup_steps)
        if step > warmup_steps:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return scale * 0.5 * (1 + np.cos(np.pi * progress))
        else:    
            return scale

    return lr_lambda_decay

def lr_lambda(step):
    scale = min(1.0, (step + 1) / 100)
    return scale

def lr_lambda_nowarm(step):
    return 1.0

## Source: https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d 
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.best_state = None
    
    def check_early_stop(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            self.best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            print(f"NO EARLY STOP, Valid Loss: {val_loss}", flush=True)
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")