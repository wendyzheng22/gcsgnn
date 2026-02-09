import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from baselines.baseline import filter_dataset
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from torch.nn.parameter import Parameter
from tqdm import tqdm
from utils import vector_to_symm_matrix


class CFExplainer(BaseModel):
    def __init__(self, pred_model: BaseModel, device: str, lr: float) -> None:
        super().__init__()
        self.pred_model = pred_model
        self.device = device
        self.lr = lr

        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Freeze weights in prediction model
        """
        for name, param in self.pred_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False

    def get_P_vec(self, dim: int) -> Parameter:
        """
        P_vec is the parameter the explainer is learning for every input graph
        Returns an initialized Parameter with a shape corresponding to a square matrix of size (dim, dim)
        """
        cur_P_vec_size = int((dim * dim + dim) / 2)

        eps = 10**-4
        with torch.no_grad():
            cur_P_vec = Parameter(
                torch.sub(torch.randn(cur_P_vec_size, device=self.device), eps),
                requires_grad=True,
            )

        return cur_P_vec

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        p_vec: Parameter,
        node_labels:bool=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of CF_GNNExplainer\n
        Returns:
            pred_output: logistic probabilities of each class
            actual_pred_output: non-differentiable but needed to get model class prediction
            threshold_p_hat: discrete symmetric matrix constructed from p_vec
        """
        num_nodes = (adj.sum(-1) > 0).sum()
        p_hat_symm_matrix = vector_to_symm_matrix(
            p_vec, num_nodes, adj.shape[0] - num_nodes
        )
        A_tilde = torch.sigmoid(p_hat_symm_matrix) * adj
        A_tilde.requires_grad_()
        pred_output = self.pred_model(
            x.unsqueeze(0), A_tilde.unsqueeze(0), edge_attr.unsqueeze(0), node_labels=node_labels
        )["graph_pred"]

        threshold_p_hat = (torch.sigmoid(p_hat_symm_matrix) >= 0.5).float()
        cf_adj = threshold_p_hat * adj
        cf_adj.requires_grad_()
        actual_pred_output = self.pred_model(
            x.unsqueeze(0), cf_adj.unsqueeze(0), edge_attr.unsqueeze(0), node_labels=node_labels
        )["graph_pred"]

        return pred_output, actual_pred_output, cf_adj

    def loss(
        self,
        log_probs: torch.Tensor,
        label: torch.Tensor,
        prediction: torch.Tensor,
        cf_adj: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # negate prediction loss to maximize the loss
        loss_pred = -F.nll_loss(log_probs, torch.tensor([label], device=self.device))
        loss_graph_dist = torch.sum(torch.abs(cf_adj - adj)) / 2

        # if label changes, zero out prediction loss
        label_prediction_matches = 1 if torch.equal(label, prediction) else 0
        loss_total = label_prediction_matches * loss_pred + 0.5 * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist

    def run_one_graph(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        label: torch.Tensor,
        epochs: int,
        node_labels:bool=True
    ) -> torch.Tensor:
        """
        Optimizes a p_vec (used to generate a counterfactual adjacency matrix) on the input graph
        """
        best_cf_adj = torch.tensor([])
        best_loss_total = np.inf
        cur_graph_nodes = (adj.sum(-1) > 0).sum()

        p_vec = self.get_P_vec(cur_graph_nodes)
        optimizer = optim.SGD([p_vec], lr=self.lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            pred_output, actual_pred_output, cf_adj = self.forward(
                x, adj, edge_attr, p_vec, node_labels
            )
            y_pred_actual = torch.argmax(actual_pred_output, dim=1)

            loss_total, _, _ = self.loss(pred_output, label, y_pred_actual, cf_adj, adj)
            loss_total.backward()
            optimizer.step()

            if y_pred_actual.item() != label and loss_total.item() < best_loss_total:
                best_cf_adj = cf_adj.detach().cpu().clone()
                best_loss_total = loss_total.item()
        return best_cf_adj

def run_cf_gnnexplainer(
    pred_model: BaseModel,
    dataset: GraphDataset,
    args: argparse.Namespace,
) -> list[Graph]:
    pred_model.eval()
    dataset = filter_dataset(dataset, pred_model, args.batch_size)
    explainer = CFExplainer(pred_model, device=args.device, lr=args.gnn_lr)
    cf_examples = []
    for graph in tqdm(dataset):
        x, adj = graph['x'].to(args.device), graph['adj_matrix'].to(args.device)
        edge_attr = graph['edge_attr_matrix'].to(args.device)
        label  = torch.tensor(graph['graph_label']).to(args.device)
        best_cf_adj = explainer.run_one_graph(x, adj, edge_attr, label, args.gnn_epochs, node_labels=dataset.node_labels)
        if len(best_cf_adj) !=0:
            cf_examples.append(Graph(
                    x=graph['x'],
                    adj_matrix=best_cf_adj.cpu(),
                    edge_attr_matrix=graph['edge_attr_matrix'],
                    graph_idx=graph['graph_idx'],
                    graph_label=int(not graph['graph_label']),
                ))
    return cf_examples
