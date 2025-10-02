import torch
import os
from tensorboardX import SummaryWriter


class Config():
    '''
    Config class
    '''
    def __init__(self):
        self.dataset_root = './data'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Improved learning parameters for pellet counting
        self.lr = 3e-5   # Even lower learning rate for numerical stability
        self.batch_size = 2  # Slightly larger batch for better gradient estimates
        self.epochs = 300    # More epochs for better convergence
        
        # Training stability parameters
        self.early_stopping_patience = 55  # Increased patience for fine-tuning
        self.lr_scheduler_patience = 12     # More patience before LR reduction
        self.lr_scheduler_factor = 0.3      # More aggressive LR reduction
        self.min_lr = 1e-7                  # Even lower minimum LR
        
        # Model parameters
        self.weight_decay = 1e-4            # Regularization strength
        self.gradient_clip_norm = 0.5       # Gradient clipping value
        
        # Paths
        self.checkpoints  = './checkpoints' # checkpoints dir
        self.writer       = SummaryWriter() # tensorboard writer

        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)