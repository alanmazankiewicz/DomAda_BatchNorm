import torch
from torch.autograd import Function
from torch.nn import Module
from torch import nn
import numpy as np
import abc

class LinForNorm(Function):
    """
    Multiplies weights (aka gamma) and adds bias (aka beta) on standardized batch.
    See also  https://pytorch.org/docs/master/notes/extending.html
    """
    
    @staticmethod
    def forward(ctx, input, weights, bias):
        ctx.save_for_backward(input, weights, bias)
        return input * weights + bias
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        grad_input = grad_weights = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weights
        if ctx.needs_input_grad[1]:
            grad_weights = (grad_output * input).sum(0)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weights, grad_bias


class Standardize(Function):
    """
    Standardize current batch with mean and variance
    See also  https://pytorch.org/docs/master/notes/extending.html
    """
    
    @staticmethod
    def forward(ctx, input, means, variances, epsilon):
        ctx.save_for_backward(input, means, variances)
        ctx.epsilon = epsilon
        output_norm = input - means
        output_norm /= (variances + epsilon).sqrt()  # save for backward und dann das mal grad output in backward
        return output_norm
    
    @staticmethod
    def backward(ctx, grad_output):
        input, means, variances = ctx.saved_tensors
        epsilon = ctx.epsilon
        N = input.size(0)
        
        X_mu = input - means
        std_inv = 1 / ((variances + epsilon).sqrt())
        
        # dx_norm = grad_output
        dvar = (grad_output * X_mu).sum(0) * -0.5 * std_inv ** 3
        dmu = (grad_output * -std_inv).sum(0) + dvar * (-2. * X_mu).mean(0)
        
        grad_input = (grad_output * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        
        return grad_input, None, None, None



class AbstractDABatchNorm(Module, abc.ABC):
    """Abstract Class"""
    
    # TODO: Pre-estimate versions actually do not need to update running stats
    
    def __init__(self, dim, domain_lst, epsilon=1e-5, momentum=0.1, safe_eval=True):
        """
        Constructor
        :param int dim:          dimensionality of input
        :param list[int]         domain_lst: List containing domain ids e.g. list(range(5)) for 5 domains named 0,1,2,3,4.
        :param float epsilon:    Value added to the denominator of the normalization for numerical stability. Default: 1e-5
        :param float momentum:   The exponential weighting factor used for the running_means and running_vars computation during training.
                                 Can be set to None for cumulative moving average (i.e. simple average). As in torch.BatchNorm Default: 0.1
        :param bool safe_eval:   Whether to run in safe evaluation mode. If true throws exception if not trained on all source domains. Default: True
        """
        if (momentum != None):
            if (momentum <= 0) | (momentum >= 1):
                raise ValueError("momentum must be in (0, 1)")
        
        super(AbstractDABatchNorm, self).__init__()
        self.target_ready = 0
        self.dim = dim
        self.epsilon = epsilon
        self.no_domains = len(domain_lst)
        self.n = [0] * self.no_domains
        self.domain_lst = sorted(domain_lst)
        self.momentum = momentum  # if used make it 0.1
        self.safe_eval = safe_eval
        self.trained_domains = np.array([0] * self.no_domains)
        self.weights = nn.Parameter(torch.ones(dim, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, requires_grad=True, dtype=torch.float32))
        self.register_buffer("running_means", torch.zeros(self.dim, dtype=torch.float32).repeat(self.no_domains, 1))
        
        if self.momentum == None:
            self.register_buffer("running_vars", self.running_means.clone())
        else:
            self.register_buffer("running_vars", torch.ones(self.dim, dtype=torch.float32).repeat(self.no_domains, 1))
        
        self.register_buffer("target_mean", torch.zeros(self.dim))
        self.register_buffer("target_var", torch.zeros(self.dim))
    
    @staticmethod
    def _inc_mean(cur_avg, prev_avg, cur_n, prev_n):
        """
        Incrementally update mean estimate given other mean and new mean over batch of size cur_n
        :param float cur_avg: mean of current batch
        :param float prev_avg: mean of all the previous batches, when first batch start with 0
        :param int cur_n: size of current batch
        :param int prev_n: sum of size of all previous batches
        :return float: new mean estimate over current and previous batches
        """
        sum_n = cur_n + prev_n
        return cur_n / sum_n * cur_avg + prev_n / sum_n * prev_avg
    
    @staticmethod
    def _inc_var(cur_s, prev_s, cur_n, prev_n, cur_avg, prev_avg):
        """
        Incrementally update variance estimate given other variance and new one over batch of size cur_n
        :param float cur_s: variance of current batch
        :param float prev_s: variance of all previous batches, when first start with 0
        :param int cur_n: size of current batch
        :param int prev_n: sum of size of all previous batches
        :param float cur_avg: mean of current batch
        :param float prev_avg: mean of all the previous batches, when first batch start with 0
        :return: new variance estimate over current and previous batches
        """
        sum_n = cur_n + prev_n
        new_s = (cur_n / sum_n) * cur_s + (prev_n / sum_n) * prev_s + (cur_n * prev_n / sum_n ** 2) * (
                prev_avg - cur_avg) ** 2
        return new_s
    
    def _norm_dom(self, real_dom):
        """Internal representation of domain ids is in range(0, range(self.domain_lst)). Returns internal id for external id."""
        return self.domain_lst.index(real_dom)
    
    def _training_phase(self, data, domain_ids):
        """
        Training pass
        :param torch.FloatTensor data: input
        :param torch.LongTensor domain_ids: 1-dim tensor containing for each instance in the batch its domain id. Must be the same id for whole batch.
        :return: standardized input by batch mean and variance
        """
        if (torch.any(domain_ids[0] != domain_ids)):
            raise ValueError("Batch passed for training contains multiple people. Make sure the dataset (features and labels!) is ordered by person_id before passing it to the dataloader.")
        
        if (domain_ids.size(0) != data.size(0)):
            raise ValueError("domain_weights must have same batch size as data")
        
        if (torch.any(domain_ids < 0)) | (torch.any(domain_ids > max(self.domain_lst))):
            raise ValueError("domain_weights at training must not contain values < 0 or > max(self.domain_lst)")
        
        real_domain = int(domain_ids[0].item())  # because of lopocv one id is always missing e.g. 0 in the first round
        domain = self._norm_dom(real_domain)  # -> "normalize" to map from real_ids to 0 (to no_people - 1)
        
        if self.safe_eval:
            self.trained_domains[domain] = 1
        
        with torch.no_grad():
            batch_mean = data.mean(0)  # detach is important for training otherwise grad funcs will accumulate and overflow GPU memory
            biased_batch_var = data.var(0, False)  # biased var estimate -> N instead of N-1. Trick to have N instead of N-1 in backward derivatives
            batch_var = data.var(0)
        
        if self.momentum == None:
            self.running_vars[domain] = self._inc_var(batch_var, self.running_vars[domain], data.size(0), self.n[domain], batch_mean, self.running_means[domain])
            self.running_means[domain] = self._inc_mean(batch_mean, self.running_means[domain], data.size(0), self.n[domain])
            self.n[domain] += data.size(0)
        else:
            self.running_vars[domain] = self.running_vars[domain] * (1 - self.momentum) + batch_var * self.momentum
            self.running_means[domain] = self.running_means[domain] * (1 - self.momentum) + batch_mean * self.momentum
        
        output = Standardize.apply(data, batch_mean, biased_batch_var, self.epsilon)
        return output
    
    def forward(self, data, domain_ids=None):
        """
        :param data: batch of input features
        :param torch.tensor domain_ids: only relevant for training. During testing = None
         contains domain_ids one for each instances in the batch. len = batch_size
         Must be the same value for whole batch e.g. batch_size = 5, domain_id = 3 -> torch.tensor([3,3,3,3,3])
         Does not have to represent ids from 0 to (no_domains -1) but can be any id -> gets normalized later

        :return: batch of same size as data, domain specific batch normalized values
        """
        
        if self.training:
            output = self._training_phase(data, domain_ids)
        
        else:  # here the domain ids in domain_weights must be w.r.t. domains in this layer, nor real domain ids
            output = self._testing_phase(data)
        
        final_output = LinForNorm.apply(output, self.weights, self.bias)
        return final_output
    
    def reset_buffers(self):
        """Resets buffers to initialization state: running_means, running_vars, n, trained_domains"""
        self.running_means.data = self.running_means.data * 0
        self.n = [0] * self.no_domains
        
        if self.momentum == None:
            self.running_vars.data = self.running_means.clone()
        else:
            self.running_vars.data = self.running_vars.data * 0 + 1
        
        self.running_vars.data = self.running_means.clone() + self.epsilon
        
        self.trained_domains = np.array([0] * self.no_domains)
    
    def init_target(self, data, *args):
        """Estimates target_mean and target_var for standardization during testing from data"""
        if self.safe_eval:
            if not np.all(self.trained_domains):
                raise Exception("Can't init target, not trained on every domain")
        
        if data.size(0) == 1:
            raise ValueError("target must not be inited with a single point")
        
        with torch.no_grad():
            mean = data.mean(0).detach()
            var = data.var(0).detach()
        
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_var", var)
        
        self.target_ready = 1
    
    @abc.abstractmethod
    def _testing_phase(self, data):
        """Testing"""
