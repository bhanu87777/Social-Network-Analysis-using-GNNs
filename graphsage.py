import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, sampling_function=None):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.sampling_function = sampling_function
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.out_conv = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if self.sampling_function:
            edge_index = self.sampling_function(edge_index)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.out_conv(x, edge_index)
        return F.log_softmax(x, dim=1)