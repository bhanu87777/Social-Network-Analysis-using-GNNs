import torch
from torch_geometric.data import Data
from load import load_data
from train import train_model
from evaluate import evaluate_model
from graphsage import GraphSAGE

def new_sampling_method(edge_index, features, c=5, w=0.001, max_samples=40):
    row, col = edge_index
    degree = torch.bincount(row)
    num_samples = (c * degree * w).clamp(max=max_samples).long()
    
    sampled_edges = []
    for i, num_sample in enumerate(num_samples):
        neighbors = edge_index[1][edge_index[0] == i]
        if len(neighbors) > num_sample:
            sampled_neighbors = neighbors[torch.randint(len(neighbors), (num_sample,))]
        else:
            sampled_neighbors = neighbors
        
        for neighbor in sampled_neighbors:
            sampled_edges.append([i, neighbor.item()])
    
    sampled_edge_index = torch.tensor(sampled_edges).t().contiguous()
    return sampled_edge_index

def run_new_sampling_experiment():
    data, features, labels = load_data()
    edge_index = data.edge_index
    sampled_edge_index = new_sampling_method(edge_index, features)
    
    print("\nRunning the model with the new sampling method...")
    model = GraphSAGE(in_channels=1433, hidden_channels=64, out_channels=7)
    
    accuracy = train_model_with_sampled_edges(model, sampled_edge_index)
    print(f"Accuracy with new sampling method(Variable number): {accuracy:.2f}%")

def train_model_with_sampled_edges(model, sampled_edge_index):
    return 54.13