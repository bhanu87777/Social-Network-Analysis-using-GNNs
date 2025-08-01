import torch
from train import train_model
from sklearn.metrics import accuracy_score

def evaluate_model():
    model, data = train_model()
    features = data.x
    labels = data.y

    def test():
        model.eval()
        with torch.no_grad():
            out = model(features, data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[data.test_mask].eq(labels[data.test_mask]).sum().item()
            accuracy = correct / data.test_mask.sum().item()
        return accuracy

    accuracy = test()
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()