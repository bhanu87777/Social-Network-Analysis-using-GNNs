import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import random
from load import load_data
from train import train_model

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def visualize_graph():
    data, _, _ = load_data()
    graph = nx.Graph()
    for edge_index in data.edge_index.T.tolist():
        graph.add_edge(edge_index[0], edge_index[1])

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, node_color=data.y.numpy(), cmap='viridis', with_labels=False, node_size=50)
    plt.title("Cora Dataset Visualization (Static Layout)")
    plt.show()

def visualize_embeddings():
    set_random_seeds()
    data, features, _ = load_data()
    model, _ = train_model()
    model.eval()
    
    with torch.no_grad():
        embeddings = model(features, data.edge_index).numpy()

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    class_names = [
        "Case-Based Reasoning (CBR)",
        "Genetic Algorithms (GA)",
        "Neural Networks (NN)",
        "Probabilistic Methods (PM)",
        "Reinforcement Learning (RL)",
        "Robotics (RO)",
        "Theory (TH)"
    ]
    labels = data.y.numpy()
    unique_classes = sorted(set(labels))
    
    palette = [
        (139/255, 0, 0),
        (0, 100/255, 0),
        (0, 0, 139/255),
        (255/255, 255/255, 0),
        (255/255, 140/255, 0),
        (255/255, 0, 255/255),
        (101/255, 67/255, 33/255)
    ]

    plt.figure(figsize=(12, 8))
    for i, class_label in enumerate(unique_classes):
        plt.scatter(
            reduced_embeddings[labels == class_label, 0],
            reduced_embeddings[labels == class_label, 1],
            label=class_names[class_label],
            s=20,
            alpha=0.7,
            color=palette[i]
        )
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title("Node Embeddings Visualization (Static & Grouped by Class)")
    plt.legend(title="Classes", loc='upper left', fontsize='small')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_random_seeds()
    visualize_graph()
    visualize_embeddings()