import torch
from torch_geometric.datasets import Planetoid

def load_data():
    dataset = Planetoid(root='cora_dataset', name='Cora')
    data = dataset[0]
    
    features = data.x
    labels = data.y
    return data, features, labels