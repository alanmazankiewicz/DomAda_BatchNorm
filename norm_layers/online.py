import torch
import numpy as np
from norm_layers.abstract import AbstractDABatchNorm, Standardize


class PreEstimateOnlineDABatchNorm(AbstractDABatchNorm):
    """Incrementally updates target person mean and var in testing phase using exponential average - initial estimate from a batch"""
    
    def __init__(self, dim, domain_lst, test_momentum, epsilon=1e-5, momentum=0.1, safe_eval=True):
        """
        Constructor
        :param int dim: dimensionality of input
        :param list[int] domain_lst: List containing domain ids e.g. list(range(5)) for 5 domains named 0,1,2,3,4.
        :param float test_momentum: Online adaptation momentum. Must be in (0,1). Weights how strong adaptation to new point / batch should be.
        :param float epsilon: Value added to the denominator of the normalization for numerical stability. Default: 1e-5
        :param float momentum: The exponential weighting factor used for the running_means and running_vars computation during training.
                               Can be set to None for cumulative moving average (i.e. simple average). As in torch.BatchNorm Default: 0.1
        :param bool safe_eval: Whether to run in safe evaluation mode. If true throws exception if not trained on all source domains. Default: True
        """
        
        if (test_momentum >= 1) | (test_momentum <= 0):
            raise ValueError("test_momentum must be in (0,1) exclusive.")
        
        super(PreEstimateOnlineDABatchNorm, self).__init__(dim, domain_lst, epsilon, momentum, safe_eval)
        self.test_momentum = test_momentum
    
    def _testing_phase(self, data):
        """If data is a single point uses incremental exponential mean and variance update. If data is batch uses batch exponential average."""
        if not self.target_ready:
            raise Exception("Target must be inited before testing phase")
        
        if len(data.size()) != 2:
            raise Exception("dimensionality of batch (or single instance) is not correct")
        
        if data.size(0) == 1:
            data_flat = data.flatten()
            self.target_var = (1 - self.test_momentum) * (
                        self.target_var + self.test_momentum * (data_flat - self.target_mean) ** 2)
            self.target_mean = (1 - self.test_momentum) * self.target_mean + self.test_momentum * data_flat
        else:
            with torch.no_grad():
                batch_mean = data.mean(0).detach()
                batch_var = data.var(0, False).detach()
            
            self.target_mean = (1 - self.test_momentum) * self.target_mean + self.test_momentum * batch_mean
            self.target_var = (1 - self.test_momentum) * self.target_var + self.test_momentum * batch_var
        
        output = Standardize.apply(data, self.target_mean, self.target_var, self.epsilon)
        
        return output


class OnlineDABatchNorm(PreEstimateOnlineDABatchNorm):
    """Incrementally updates target person mean and var in testing phase using exponential average - initial estimate as average of running stats from training"""
    
    def init_target(self, *args):
        """Init target stats as mean of running stats. No arguments."""
        if self.safe_eval:
            if not np.all(self.trained_domains):
                raise Exception("Can't init target, not trained on every domain")
            
            mean = self.running_means.mean(0).detach()
            var = self.running_vars.mean(0).detach()
            
            self.register_buffer("target_mean", mean)
            self.register_buffer("target_var", var)
            
            self.target_ready = 1