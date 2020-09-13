from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))
import torch
import numpy as np
from norm_layers.static import StaticDABatchNorm as BatchNorm
from sampler.sampler import DomainBatchSampler as MySampler
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from torch.nn import Module
from torch import nn
from torch.nn import BatchNorm1d
import random
from norm_layers.online import OnlineDABatchNorm


class TestDataset(Dataset):
    
    def __init__(self, X, y):
        super(TestDataset, self).__init__()
        self.X = X
        self.cl = y[:, 0].view(-1, 1)
        self.domain = y[:, 1].view(-1, 1)
    
    def __getitem__(self, index):
        return (self.X[index], self.cl[index], self.domain[index])
    
    def __len__(self):
        return self.X.size(0)


def create_test_data(three_class = False, dataset_class= TestDataset):
    """Creates test_data from iris, 2 classes to persons, sorted by persons, 50 instances per person"""
    iris = load_iris()
    x_data = iris.data
    y_data = iris.target
    
    if three_class:
        n_instances = 150
        repeats = 3
        batch_size = 5
    else:
        n_instances = 100
        repeats = 2
        batch_size = 10
    
    x_data = torch.Tensor(x_data[0:n_instances, :])
    y_data = torch.Tensor(y_data[0:n_instances]).view(-1, 1)
    
    person = torch.cat([torch.tensor([0], dtype=torch.float32).repeat(25), torch.tensor([1], dtype=torch.float32).repeat(25)]).repeat(repeats).view(-1, 1)
    y_data = torch.cat([y_data, person], 1)
    y_data = y_data[y_data[:, 1].sort()[1]]
    x_data = x_data[y_data[:, 1].sort()[1]]
    
    cov_shift = [10, -10, 30]
    for j,i in enumerate(range(0, n_instances, 50)):
        x_data[i:i+50] = x_data[i:i+50] + cov_shift[j]

    sampler = MySampler(int(n_instances / 2), 2, batch_size)
    dataset = dataset_class(x_data, y_data)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return (dataloader, x_data, y_data)

def test_online():
    test_momentum = 0.1
    momentum = 0.1
    dim = 10
    people_lst = [0, 1]
    epsilon = 1e-5
    no_instances = 6
    batchnorm = OnlineDABatchNorm(dim, people_lst, test_momentum, epsilon, momentum)

    batch_1 = torch.arange(no_instances * dim).view(no_instances, dim).float()
    batch_2 = batch_1 * 2
    target_batch = batch_1 * 4
    dom_1 = torch.zeros(6)
    dom_2 = torch.ones(6)

    _ = batchnorm(batch_1, dom_1)
    _ = batchnorm(batch_2, dom_2)

    true_out = (target_batch - target_batch.mean(0)) / (target_batch.var(0, False) + 1e-5).sqrt()

    batchnorm = batchnorm.eval()
    batchnorm.init_target()

    for i in range(1000):
        out = batchnorm(target_batch, None)

    assert torch.allclose(batchnorm.target_mean, target_batch.mean(0)), "target_mean does not converge towards true in batch case"
    assert torch.allclose(batchnorm.target_var, target_batch.var(0, False)), "target_var does not converge towards true in batch case"
    assert torch.allclose(true_out, out), "target output does not converge towards true in batch case"

    # Test single point
    # batchnorm.target_ready = 0
    batchnorm.test_momentum = 0.001

    for i in range(5000):
        for instance in target_batch:
            instance = instance.view(1, dim)
            out = batchnorm(instance, None)
        target_batch = target_batch[torch.randperm(no_instances)]

    assert torch.allclose(batchnorm.target_mean, target_batch.mean(0), atol=0.1), "target_mean does not converge towards true in single point case"
    assert torch.allclose(batchnorm.target_var, target_batch.var(0, False), atol=0.1), "target_mean does not converge towards true in single point case"

def test_expo_avg():
    norm = BatchNorm(10, [0,1], momentum=0.1, safe_eval=False)
    tor_norm = BatchNorm1d(10)
    inp = torch.FloatTensor(range(10)).repeat(10).view(10, 10).t()
    domain_0 = torch.zeros(10, dtype=torch.int).view(-1, 1)
    
    for i in range(2, 100):
        inp = inp * 2
        a = norm(inp, domain_0)
        a = tor_norm(inp)
        
        assert torch.allclose(norm.running_means[0], tor_norm.running_mean), "exponential running mean failed"
        assert torch.allclose(norm.running_vars[0], tor_norm.running_var, rtol= 0.05), "exponential running var failed"

def test_inc_func():
    """
    Tests if inc stat computation is correct
    """
    test_data = np.random.randint(10, size=(100, 5))
    test_data_1 = torch.tensor(test_data, dtype=torch.float32)
    
    n_digits = 3
    true_mean = torch.round(test_data_1.mean(0) * 10**n_digits) / (10**n_digits)
    true_var = torch.round(test_data_1.var(0, False) * 10**n_digits) / (10**n_digits)
    
    np.random.shuffle(test_data)
    test_data_1 = torch.tensor(test_data, dtype=torch.float32)
    
    n = 0
    prev_mean = 0
    prev_var = 0
    
    for i in range(0, 100, 25):
        batch = test_data_1[i:i + 25]
        prev_var = BatchNorm._inc_var(batch.var(0, False), prev_var, 25, n, batch.mean(0), prev_mean)
        prev_mean = BatchNorm._inc_mean(batch.mean(0), prev_mean, 25, n)
        n += 25
    
    prev_mean = torch.round(prev_mean * 10**n_digits) / (10**n_digits)
    prev_var = torch.round(prev_var * 10**n_digits) / (10**n_digits)
    
    assert torch.all(true_mean == prev_mean), "inc_mean test failed"
    assert torch.all(torch.isclose(true_var, prev_var, atol = 1e-2)), "inc_var test failed"
    
    
    
def test_running_stats():
    """Tests if running stats computed over iterations are correct"""
    def main_flow(momentum):
        norm = BatchNorm(10, [0, 1], momentum=momentum, safe_eval=True)
        tor_norm_1 = BatchNorm1d(10, momentum=momentum)

        tor_norm_1.weight.data = norm.weights.data.clone()
        tor_norm_1.bias.data = norm.bias.data.clone()

        inp = torch.FloatTensor(range(5)).repeat(10).view(10, 5).t()
        inp2 = inp * 2
        domain_0 = torch.zeros(5, dtype=torch.int).view(-1, 1)
        domain_1 = torch.ones(5, dtype=torch.int).view(-1, 1)

        norm(inp, domain_0)
        norm(inp, domain_1)
        tor_norm_1(inp)

        norm(inp2, domain_0)
        norm(inp2, domain_1)
        tor_norm_1(inp2)

        assert torch.allclose(norm.running_means[0], tor_norm_1.running_mean), "Running mean failed on "  + str(momentum)
        assert torch.allclose(norm.running_means[1], tor_norm_1.running_mean), "Running mean failed on "  + str(momentum)
        
        if momentum == None:
            assert np.all(np.concatenate([np.array(inp), np.array(inp2)]).var(0) == norm.running_vars[0].numpy())
            assert np.all(np.concatenate([np.array(inp), np.array(inp2)]).var(0) == norm.running_vars[1].numpy())
        else:
            assert torch.allclose(norm.running_vars[0], tor_norm_1.running_var), "Running var failed on " + str(momentum)
            assert torch.allclose(norm.running_vars[1], tor_norm_1.running_var), "Running var failed on " + str(momentum)

    for i in range(1, 10):
        main_flow(round(i*0.1, 2))
    # main_flow(None)


def test_change_in_weights():
    """Tests if weights change in linear layer when computing gradients through the BN layer in NN """
    dataloader, x_data, y_data = create_test_data()

    random.seed(2)
    torch.manual_seed(2)
    np.random.seed(2)
    
    class Net_1(Module):
        
        def __init__(self):
            super(Net_1, self).__init__()
            self.linear1 = nn.Linear(4, 2)
            self.norm = BatchNorm(2, [0,1], momentum=None, safe_eval=False)
            self.linear2 = nn.Linear(2, 1)
            
        
        def forward(self, input, domain_weights):
            output = self.norm(self.linear1(input), domain_weights)
            output = torch.sigmoid(self.linear2(output))
            return output
    
    net_1 = Net_1()
    net_1 = net_1.train()
    criterion = torch.nn.MSELoss()
    optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=2)
    
    results_1 = []
    results_2 = []
    results_3 = []
    for epoch in range(2):
        for X, label, domain in dataloader:
            optimizer_1.zero_grad()
            output = net_1(X, domain)
            loss = criterion(output, label)
            loss.backward()
            optimizer_1.step()
            
        results_1.append(net_1.linear1.weight.clone())
        results_2.append(net_1.linear2.weight.clone())
        results_3.append(net_1.linear1.bias.clone())
        
    
    assert torch.any(results_2[0] != results_2[1]), "Weight updates dont flow from error to last layer"
    assert torch.any(results_1[0] != results_1[1]), "Weight updates dont flow through batchnorm"
    assert torch.any(results_3[0] != results_3[1]), "Bias updates dont flow through batchnorm"



def test_change_in_weights_3_out():
    """Tests if weights change in linear layer when computing gradients through the BN layer in NN having 3 outputs"""
    dataloader, x_data, y_data = create_test_data(True)
    
    class Net_1(Module):
        
        def __init__(self):
            super(Net_1, self).__init__()
            self.linear1 = nn.Linear(4, 3, bias=True)
            self.norm = BatchNorm(3, [0,1], momentum=None, safe_eval=False)
            self.linear2 = nn.Linear(3, 3, bias = True)
        
        def forward(self, input, domain_weights):
            output = self.norm(self.linear1(input), domain_weights)
            output = torch.softmax(self.linear2(output), 0)
            return output
    
    net_1 = Net_1()
    net_1 = net_1.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=0.1)
    
    results_1 = []
    results_2 = []
    for epoch in range(2):
        for X, label, domain in dataloader:
            optimizer_1.zero_grad()
            output = net_1(X, domain)
            loss = criterion(output, label.long().view(5))
            loss.backward()
            optimizer_1.step()
        
        results_1.append(net_1.linear1.weight.clone())
        results_2.append(net_1.linear2.weight.clone())
    
    assert torch.any(results_2[0] != results_2[1]), "Weight updates dont flow from error to last layer"
    assert torch.any(results_1[0] != results_1[1]), "Weight updates dont flow through batchnorm"
    

def test_leaf_grad():
    """
    Tests if grads can be computed to w.r.t. gamma and beta
    """
    
    norm_layer = BatchNorm(10, [0,1], momentum=None, safe_eval=False)
    assert norm_layer.weights.is_leaf, "weight.is_leaf failed"
    assert norm_layer.bias.is_leaf, "bias.is_leaf failed"
    assert norm_layer.weights.requires_grad, "weight.requires_grad failed"
    assert norm_layer.bias.requires_grad, "bias.requires_grad failed"
    
    
def test_output():
    """Tests output during testing and training with torch.BatchNorm"""
    def main_flow(momentum):
        norm = BatchNorm(10, [0, 1], momentum=momentum, safe_eval=False)
        tor_norm_1 = BatchNorm1d(10, momentum=momentum)
        tor_norm_2 = BatchNorm1d(10, momentum=momentum)
        tor_norm_1.weight.data = norm.weights.data.clone()
        tor_norm_1.bias.data = norm.bias.data.clone()
        tor_norm_2.weight.data = norm.weights.data.clone()
        tor_norm_2.bias.data = norm.bias.data.clone()

        inp = torch.FloatTensor(range(5)).repeat(10).view(10, 5).t()
        inp2 = inp * 3
        domain_0 = torch.zeros(5, dtype=torch.int).view(-1, 1)
        domain_1 = torch.ones(5, dtype=torch.int).view(-1, 1)

        domain_0_eval = torch.FloatTensor([1, 0]).repeat(5).view(5, 2)
        domain_1_eval = torch.FloatTensor([0, 1]).repeat(5).view(5, 2)

        a = norm(inp, domain_0)
        b = tor_norm_1(inp)
        assert torch.allclose(a, b), "Output during train not the same as torch.BatchNorm output on " + str(momentum)

        a = norm(inp2, domain_1)
        b = tor_norm_2(inp2)
        assert torch.allclose(a, b), "Output during train not the same as torch.BatchNorm output on " + str(momentum)

        norm = norm.eval()  # swicth train and test data
        tor_norm_1 = tor_norm_1.eval()
        tor_norm_2 = tor_norm_2.eval()
        
        norm.target_ready = 1
        norm.target_mean  = norm.running_means[1]
        norm.target_var = norm.running_vars[1]

        a = norm(inp, domain_1_eval)
        b = tor_norm_2(inp)
        assert torch.allclose(a, b), "Output during test not the same as torch.BatchNorm output on " + str(momentum)

        norm.target_mean = norm.running_means[0]
        norm.target_var = norm.running_vars[0]
        a = norm(inp2, domain_0_eval)
        b = tor_norm_1(inp2)
        assert torch.allclose(a, b), "Output during test not the same as torch.BatchNorm output on " + str(momentum)
        
    for i in range(1, 10):
        main_flow(round(i*0.1, 2))
    main_flow(None)
    
def test_grad():
    """tests if grads are the same between my implementation and torch"""
    momentum = 0.1
    norm = BatchNorm(10, [0, 1], momentum=momentum, safe_eval=False)
    tor_norm_1 = BatchNorm1d(10, momentum=momentum)
    tor_norm_1.weight.data = norm.weights.data.clone()
    tor_norm_1.bias.data = norm.bias.data.clone()
    inp = torch.rand(5, 10)
    domain_0 = torch.zeros(5, dtype=torch.int).view(-1, 1)
    a = norm(inp, domain_0)
    b = tor_norm_1(inp)
    soft = torch.nn.Softmax(1)
    crit = torch.nn.CrossEntropyLoss()
    loss_a = crit(soft(a), torch.ones(5, dtype=torch.long))
    loss_b = crit(soft(b), torch.ones(5, dtype=torch.long))
    loss_a.backward()
    loss_b.backward()
    assert torch.allclose(norm.weights.grad, tor_norm_1.weight.grad), "Grads between torch and my batchnorm are not same"
    
def test_backward():
    """Tests weather backward works correctly"""
    iris = load_iris()
    x_data = torch.FloatTensor(iris.data)[0:100]
    y_data = torch.LongTensor(iris.target)[0:100]
    domain = torch.zeros(100).long()

    def one_hot(y, num_labels):
        return (torch.arange(num_labels) == y[:, None]).float()

    def get_accuracy(model, feat, lab, weights=None):
        """Compute accuracy, feat being the features, lab being the labels as torch.tensors"""
        model = model.eval()
        with torch.no_grad():
            # feat, lab = data.X, data.y
            if type(weights) != type(None):
                output = model(feat, weights)
            else:
                output = model(feat)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == lab).sum().item()
            total = lab.size(0)
            accur = round(correct / total, 4)
            return accur

    class Net_1(Module):
    
        def __init__(self):
            super(Net_1, self).__init__()
            self.linear1 = nn.Linear(4, 100)
            self.norm = BatchNorm(100, [0, 1], momentum=0.1, safe_eval=False)
            self.linear2 = nn.Linear(100, 2)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(1)
    
        def forward(self, input, domain_weights=None):
            output = self.relu(self.norm(self.linear1(input), domain_weights))
            output = self.softmax(self.linear2(output))
            return output

    class Net_2(Module):
    
        def __init__(self):
            super(Net_2, self).__init__()
            self.linear1 = nn.Linear(4, 100)
            self.norm = torch.nn.BatchNorm1d(100)
            self.linear2 = nn.Linear(100, 2)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(1)
    
        def forward(self, input):
            output = self.relu(self.norm(self.linear1(input)))
            output = self.linear2(output)
            output = self.softmax(output)
            return output

    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)

    net_1 = Net_1()
    net_1 = net_1.train()

    net_2 = Net_2()
    net_2 = net_2.train()
    criterion = torch.nn.CrossEntropyLoss()

    net_2.linear1.weight.data = net_1.linear1.weight.data.clone()
    net_2.linear1.bias.data = net_1.linear1.bias.data.clone()
    net_2.linear2.weight.data = net_1.linear2.weight.data.clone()
    net_2.linear2.bias.data = net_1.linear2.bias.data.clone()

    net_2.norm.weight.data = net_1.norm.weights.data.clone()
    net_2.norm.bias.data = net_1.norm.bias.data.clone()

    optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=0.1, weight_decay=0.001)
    optimizer_2 = torch.optim.SGD(net_2.parameters(), lr=0.1, weight_decay=0.001)

    for epoch in range(1000):
        optimizer_1.zero_grad()
        output = net_1(x_data, domain)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer_1.step()
        
    loss_1 = loss.item()
    net_1.norm.target_ready = 1
    net_1.norm.target_mean = net_1.norm.running_means[0]
    net_1.norm.target_var = net_1.norm.running_vars[0]
    accur1 = get_accuracy(net_1, x_data, y_data)

    for epoch in range(1000):
        optimizer_2.zero_grad()
        output = net_2(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer_2.step()
        
    loss_2 = loss.item()
    accur2 = get_accuracy(net_2, x_data, y_data)
    
    assert loss_1 == loss_2, "test_backward faild on loss. loss_1, loss_2 = " + str(loss_1) + " " + str(loss_2)
    assert accur1 == accur2,  "test_backward faild on accuracy. accur1, accur2 = " + str(accur1) + " " + str(accur1)
    
    assert torch.allclose(net_1.linear1.weight, net_2.linear1.weight, atol=1e-6), "test_backward failed on linear1.weight"
    assert torch.allclose(net_1.linear1.weight.grad, net_2.linear1.weight.grad, atol=1e-4), "test_backward failed on linear1.weight.grad"
    assert torch.allclose(net_1.norm.weights, net_2.norm.weight, atol=1e-6), "test_backward failed on norm.weight"
    assert torch.allclose(net_1.norm.weights.grad, net_2.norm.weight.grad, atol=1e-6), "test_backward failed on norm.weight.grad"
    assert torch.allclose(net_1.norm.bias.grad, net_2.norm.bias.grad, atol=1e-4), "test_backward failed on norm.bias.grad"

if __name__ == "__main__":
    print("")
    print("Testing batchnorm...")
    test_leaf_grad()
    test_inc_func()
    test_running_stats()
    test_change_in_weights()
    test_change_in_weights_3_out()
    test_expo_avg()
    test_output()
    test_grad()
    test_backward()
    test_online()
    print("Tesed batchnorm successfully")