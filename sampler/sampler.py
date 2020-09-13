import torch
from torch.utils.data.sampler import  Sampler

class DomainBatchSampler(Sampler):
	"""
	Torch custom sampler. Pass to DataLoader for getting batches. At each epoch order of instances within domain gets
	randomized but preserved between domains. Batches with instances of the same domain only get created and randomized
	in their order of passing during each epoch.
	
	Attention: The sampler assumes that each domain has the exact same no of instances and that the instances are all ordered
	according to the domain_ids (all instances of one domain followed by all instances of the next domain etc.).
	
	IMPORTANT: size_per_domain must be multiple of batch_size! Choose batch_size accordingly.
	"""
	
	def __init__(self, size_per_domain, n_domains, batch_size):
		"""
		
		:param int size_per_domain: No of instances per domain
		:param int n_domains: No. of domains
		:param int batch_size: No of instances per batch
		"""
		
		if size_per_domain % batch_size != 0:
			raise ValueError("size_per_domain must be multiple of batch_size. Choose batch_size accordingly.")
		
		self.size_per_domain = size_per_domain
		self.n_domains = n_domains
		self.batch_size = batch_size
	
	def __shuffle_within_person(self, size_per_domain, n_domains):
		container = []
		for i in range(n_domains):
			tmp = torch.randperm(size_per_domain) + i * size_per_domain
			container.append(tmp)
		
		return torch.cat(container, 0)
	
	def __shuffle_within_batches(self, size_per_domain, n_domains, batch_size):
		no_of_batches = int((size_per_domain * n_domains) / batch_size)
		return torch.randperm(no_of_batches)
	
	def __iter__(self):
		low_lvl_idx = self.__shuffle_within_person(self.size_per_domain, self.n_domains)
		mid_lvl_idx = self.__shuffle_within_batches(self.size_per_domain, self.n_domains, self.batch_size)
		
		for batch in mid_lvl_idx:
			start = batch * self.batch_size
			end = start + self.batch_size
			idx = low_lvl_idx[start: end]
			yield idx
	
	def __len__(self):
		return self.size_per_domain * self.n_domains
			
