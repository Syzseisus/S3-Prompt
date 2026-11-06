import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import GINConv, global_add_pool


class SharedGNN(nn.Module):
    def __init__(self, hidden_dim=300):
        super().__init__()
        # OGB Atom Encoder (Input dim is handled internally by OGB)
        self.atom_encoder = AtomEncoder(hidden_dim)

        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        )
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        )
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        )

    def forward(self, x, edge_index, batch):
        # x is [num_nodes, 9] (LongTensor from OGB)
        h = self.atom_encoder(x)  # -> [num_nodes, hidden_dim]
        h = self.conv1(h, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.conv3(h, edge_index)
        h = torch.relu(h)
        return global_add_pool(h, batch)
