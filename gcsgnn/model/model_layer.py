import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: int,
        dropout:float = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels  # size of the input samples
        self.out_channels = out_channels  # size of the output embeddings
        self.edge_attr_dim = edge_attr_dim
        self.dropout = dropout

        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(self.edge_attr_dim, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.att = nn.Parameter(torch.empty(1, 1, out_channels))
        
        self.init_parameters()

    def init_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin_l.weight)
        bound = 1.0 / math.sqrt(self.in_channels)
        torch.nn.init.uniform_(self.lin_l.bias, -bound, bound)

        torch.nn.init.xavier_uniform_(self.lin_r.weight)
        bound = 1.0 / math.sqrt(self.in_channels)
        torch.nn.init.uniform_(self.lin_r.bias, -bound, bound)

        torch.nn.init.xavier_uniform_(self.lin_edge.weight)
        torch.nn.init.xavier_uniform_(self.att)
            
        stdv = math.sqrt(6.0 / (self.att.size(-2) + self.att.size(-1)))
        self.att.data.uniform_(-stdv, stdv)
    
    def add_edge_attr_diagonal(self, adj:torch.Tensor, edge_attr:torch.Tensor) -> torch.Tensor:
        if len(edge_attr.shape) == 4 and edge_attr.shape[-1] == self.edge_attr_dim:
            edge_attr = edge_attr.argmax(-1)
        B, N, _ = edge_attr.shape
        num_edges = adj.sum(-1)
        num_edges[num_edges == 0] = 1
        diag = (edge_attr.sum(-1) / num_edges).unsqueeze(-1)
        diag[diag > (self.edge_attr_dim - 1)] = (self.edge_attr_dim - 1)
        full_diag = torch.eye(N).repeat(B, 1, 1).to(edge_attr.device) * diag
        edge_attr = edge_attr + torch.eye(N).repeat(B, 1, 1).to(edge_attr.device) * full_diag
        return F.one_hot(edge_attr.long(), num_classes=self.edge_attr_dim).float()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        node_labels: bool = True,
        pool: bool = True
    ) -> torch.Tensor:
        """
        Params:
            x: list of node features (shape = (B, N))
            adj: list of adjacency matrices (shape = (B, N, N))
            edge_attr: list of edge attribute matrices (shape = (B, N, N))

            B = number of instances in batch
            N = max number of nodes
        """
        if len(x.shape) == 2 and node_labels:
            x = F.one_hot(x.long(), num_classes=self.in_channels).float()
        B, N, _ = x.shape
        x_l = self.lin_l(x).unsqueeze(2)
        x_r = self.lin_r(x).unsqueeze(1)
        x = x_l + x_r

        edge_attr = self.add_edge_attr_diagonal(adj, edge_attr)
        edge_attr = self.lin_edge(edge_attr)
        x = x + edge_attr
        adj = adj + torch.eye(N, device=adj.device).repeat(B, 1, 1)
        x = (F.leaky_relu(x, 0.2) * self.att).sum(dim=-1)
        x = x.masked_fill(adj == 0, float('-inf'))
        x_num = (x - x.max(-2)[0].unsqueeze(-1)).exp()
        alpha = x_num / x_num.sum(-2).unsqueeze(-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = (alpha.unsqueeze(-1) * x_l).sum(-2) + self.bias
        if pool:
            out = out.mean(1)
        return out


class DecodeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        node_feat_dim: int,
        edge_attr_dim: int,
        max_num_nodes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels  # size of the input embeddings
        self.node_feat_dim = node_feat_dim
        self.edge_attr_dim = edge_attr_dim
        self.max_num_nodes = max_num_nodes
        self.dropout = dropout
        self.emb_exp = nn.Linear(in_channels, max_num_nodes * in_channels)
        self.node_edge_sep = nn.Linear(in_channels, in_channels)
        self.x_decoder = nn.Linear(in_channels, node_feat_dim)
        self.edge_attr_shape = int(max_num_nodes * (max_num_nodes - 1) / 2)
        self.edge_decoder = nn.Linear(max_num_nodes * in_channels, self.edge_attr_shape * self.edge_attr_dim)
        self.init_parameters()

    def init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear | nn.BatchNorm1d)):
                m.reset_parameters()

    def forward(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = embeddings.shape[0]
        exp_embs = self.emb_exp(embeddings).view(B, self.max_num_nodes, self.in_channels)
        exp_embs = F.layer_norm(F.leaky_relu(exp_embs, negative_slope=0.2), normalized_shape=exp_embs.shape)
        node_emb = self.node_edge_sep(exp_embs)
        edge_emb = (exp_embs - node_emb).view(B, -1)
        node_emb = F.layer_norm(F.dropout(F.leaky_relu(node_emb, negative_slope=0.2), p=self.dropout, training=self.training), normalized_shape=node_emb.shape)
        edge_emb = F.layer_norm(F.dropout(F.leaky_relu(edge_emb, negative_slope=0.2), p=self.dropout, training=self.training), normalized_shape=edge_emb.shape)
        x = self.x_decoder(node_emb)
        edge_attr_vals = self.edge_decoder(edge_emb).view(B, self.edge_attr_shape, self.edge_attr_dim)
        edge_attr = torch.zeros(
            B, self.max_num_nodes, self.max_num_nodes, self.edge_attr_dim
        ).to(embeddings.device)
        indices = torch.tril_indices(
            self.max_num_nodes, self.max_num_nodes, offset=-1
        ).to(embeddings.device)
        edge_attr[:, indices[0], indices[1]] = edge_attr_vals
        edge_attr = edge_attr + edge_attr.transpose(1, 2)
        adj = (1 - F.sigmoid(edge_attr)[:, :, :, 0]) 
        return x, adj, edge_attr
