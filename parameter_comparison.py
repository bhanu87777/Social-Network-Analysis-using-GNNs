import time
import torch
from graphsage import GraphSAGE
from load import load_data
from torch_geometric.transforms import RandomLinkSplit

def train_model_with_params(K, S1, S2, S3):
    data, features, labels = load_data()
    transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data)

    edge_index = train_data.edge_index.to(torch.long)
    num_classes = len(torch.unique(data.y))

    model = GraphSAGE(in_channels=1433, hidden_channels=S1, 
                      out_channels=num_classes, num_layers=K)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = criterion(out[train_data.train_mask], labels[train_data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    start_time = time.time()

    for epoch in range(100):
        loss = train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    model.eval()
    out = model(features, edge_index)
    _, predicted = out[test_data.test_mask].max(dim=1)
    correct = (predicted == labels[test_data.test_mask]).sum().item()
    accuracy = correct / test_data.test_mask.sum().item() * 100

    running_time = time.time() - start_time

    return accuracy, running_time

def run_experiment():
    K_values = [2, 3]
    S1_values = [20, 25]
    S2_values = [10, 20]
    S3_values = [0, 10]

    results = []

    for K in K_values:
        for S1 in S1_values:
            for S2 in S2_values:
                for S3 in S3_values:
                    accuracy, running_time = train_model_with_params(K, S1, S2, S3)
                    results.append((K, S1, S2, S3, accuracy, running_time))

    print("\n--- Experiment Results ---")
    print("K\tS1\tS2\tS3\tAccuracy\tRunning Time (s)")

    for result in results:
        K, S1, S2, S3, accuracy, running_time = result
        print(f"{K}\t{S1}\t{S2}\t{S3}\t{accuracy:.2f}%\t{running_time:.2f}")