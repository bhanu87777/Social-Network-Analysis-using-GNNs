from torch_geometric.datasets import Planetoid
import pandas as pd

dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

edges_df = pd.DataFrame(data.edge_index.T.numpy(), columns=['Source', 'Target'])

features_df = pd.DataFrame(data.x.numpy())
features_df['Label'] = data.y.numpy()

edges_df.to_csv('cora_edges.csv', index=False)
features_df.to_csv('cora_features_labels.csv', index=False)