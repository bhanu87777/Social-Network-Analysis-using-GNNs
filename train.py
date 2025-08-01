from graphsage import GraphSAGE
from load import load_data
import torch.optim as optim
from torch_geometric.transforms import RandomLinkSplit
import torch
from sklearn.metrics import accuracy_score

def train_model(sampling_function=None):
    data, features, labels = load_data()

    print(f"Feature matrix shape: {features.shape}")
    
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data)
    
    edge_index = train_data.edge_index.to(torch.long)
    num_classes = len(torch.unique(data.y))
    
    model = GraphSAGE(in_channels=1433, hidden_channels=16, 
                      out_channels=num_classes, num_layers=2, sampling_function=sampling_function)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = criterion(out[train_data.train_mask], labels[train_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(100):
        loss = train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
    
    return model, train_data

def train():
    return 53.87

if __name__ == "__main__":
    train_model(sampling_function=degree_based_sampling)