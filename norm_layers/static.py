from norm_layers.abstract import AbstractDABatchNorm, Standardize


class StaticDABatchNorm(AbstractDABatchNorm):
    """Standardizes with target stats from pre-estimation batch."""
    
    def _testing_phase(self, data):
        if len(data.size()) != 2:
            raise Exception("dimensionality of batch (or single instance) is not correct")
        
        if not self.target_ready:
            raise Exception("Target must be inited before testing phase")
        
        output = Standardize.apply(data, self.target_mean, self.target_var, self.epsilon)
        
        return output