import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baselines.baseline import filter_dataset
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from tqdm import tqdm


class GNNExplainer(BaseModel):
    def __init__(
        self,
        pred_model: BaseModel,
        device: str,
        lr: float,
        node_dim: int,
        edge_dim: int,
    ) -> None:
        super().__init__()
        self.pred_model = pred_model
        self.device = device
        self.lr = lr
        self.node_feat_dim = node_dim
        self.edge_attr_dim = edge_dim

        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Freeze weights in prediction model
        """
        for name, param in self.pred_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

    def get_masks(
        self, node_dim: int, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_mask = nn.Parameter(
            torch.zeros(node_dim, device=self.device), requires_grad=True
        )
        edge_mask = nn.Parameter(
            torch.empty(num_nodes, num_nodes).to(self.device), requires_grad=True
        )
        nn.init.xavier_normal_(edge_mask)
        return x_mask, edge_mask

    def get_masked_adj(self, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sym_mask = torch.sigmoid(mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        masked_adj = adj * sym_mask
        diag_mask = (
            torch.ones(adj.shape[1], adj.shape[1]) - torch.eye(adj.shape[1])
        ).to(self.device)
        return masked_adj * diag_mask

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        x_mask: torch.Tensor,
        node_labels:bool=True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_adj = self.get_masked_adj(adj, edge_mask)
        x = x * torch.sigmoid(x_mask)
        ypred = self.pred_model(
            x.unsqueeze(0), masked_adj.unsqueeze(0), edge_attr.unsqueeze(0), node_labels=node_labels
        )["graph_pred"].squeeze()
        return ypred, masked_adj

    def loss(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        edge_mask: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: prediction made by current model
            label: the label predicted by the original model.
        """
        pred_loss = (
            1000 * -pred[1 - label]
        )  # want to maximize prediction of the counterfactual
        binary_mask = torch.sigmoid(edge_mask)
        size_loss = 0.005 * binary_mask.sum()
        x_mask = torch.sigmoid(x_mask)
        mask_ent = -binary_mask * torch.log(binary_mask) - (
            1 - binary_mask
        ) * torch.log(1 - binary_mask)
        return pred_loss + size_loss + mask_ent.mean() + x_mask.mean()

    def run_one_graph(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        label: torch.Tensor,
        epochs: int,
        node_labels:bool = True
    ) -> torch.Tensor:
        best_adj = torch.tensor([])
        best_loss_total = np.inf
        
        if node_labels:
            x = F.one_hot(
                x.squeeze().long(), num_classes=self.node_feat_dim
            ).float()
        edge_one_hot = F.one_hot(
            edge_attr.long(), num_classes=self.edge_attr_dim
        ).float()

        x_mask, edge_mask = self.get_masks(x.shape[-1], adj.shape[-1])
        optimizer = optim.Adam([edge_mask, x_mask], lr=self.lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            ypred, masked_adj = self(x, adj, edge_one_hot, edge_mask, x_mask, node_labels=node_labels)
            loss_total = self.loss(ypred, label, edge_mask, x_mask)
            loss_total.backward()
            optimizer.step()

            if ypred.argmax() != label and loss_total.item() < best_loss_total:
                best_adj = (masked_adj * adj).detach().cpu().clone()
                best_loss_total = loss_total.item()
        return best_adj

def run_gnn_explainer(
    pred_model: BaseModel,
    dataset: GraphDataset, 
    args: argparse.Namespace,
) -> list[Graph]:
    pred_model.eval()
    dataset = filter_dataset(dataset, pred_model, args.batch_size)
    explainer = GNNExplainer(
        pred_model,
        device=args.device,
        lr=args.gnn_lr,
        node_dim=pred_model.node_feat_dim,
        edge_dim=pred_model.edge_attr_dim,
    )
    gnn_examples = []
    for graph in tqdm(dataset):
        x, adj = graph['x'].to(args.device), graph['adj_matrix'].to(args.device)
        edge_attr = graph['edge_attr_matrix'].to(args.device)
        label  = torch.tensor(graph['graph_label']).to(args.device)
        best_cf_adj = explainer.run_one_graph(x, adj, edge_attr, label, args.gnn_epochs, node_labels=dataset.node_labels)
        if best_cf_adj.shape[0] != 0:
            g = Graph(
                x=graph['x'],
                adj_matrix=best_cf_adj.cpu(),
                graph_idx=graph['graph_idx'],
                graph_label=int(not graph['graph_label']),
            )
            g['edge_attr_matrix'] = graph['edge_attr_matrix']
            gnn_examples.append(g)
    return gnn_examples
