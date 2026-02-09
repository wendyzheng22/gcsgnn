import argparse
import math
import os
import random
import time
from functools import partial

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from data.data_utils import get_dataloaders
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from model.model_layer import GATv2Layer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

class GATNet(BaseModel):
    def __init__(
        self,
        num_features: int,
        embedding_dim: int,
        num_classes: int,
        num_edge_attr: int,
        max_num_nodes: int,
        device: str,
        undesired_label: int,
        dropout: float = 0.6,
        prototypes_per_class: int = 5,
        num_gnn_layers: int = 1,
    ):
        super().__init__()
        self.device = device
        self.max_num_nodes = max_num_nodes
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.node_feat_dim = num_features
        self.edge_attr_dim = num_edge_attr
        self.undesired_label = undesired_label
        self.prototypes_per_class = prototypes_per_class

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(
            GATv2Layer(num_features, embedding_dim, num_edge_attr, dropout)
        )
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(
                GATv2Layer(embedding_dim, embedding_dim, num_edge_attr, dropout)
            )
        self.gnn_non_linear = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.Softmax = nn.Softmax(dim=-1)

        # prototype layers
        self.epsilon = 1e-4
        self.prototype_shape = (num_classes * prototypes_per_class, embedding_dim)
        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )
        self.num_prototypes = self.prototype_shape[0]
        self.last_layer = nn.Linear(
            self.num_prototypes, num_classes, bias=False
        )  # do not use bias
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes, num_classes
        ).to(device)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // prototypes_per_class] = 1
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear | nn.BatchNorm1d)):
                m.reset_parameters()

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = (
            -2 * xp
            + torch.sum(x**2, dim=1, keepdim=True)
            + torch.t(torch.sum(self.prototype_vectors**2, dim=1, keepdim=True))
        )
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        node_labels: bool = True,
    ):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, adj, edge_attr, node_labels)
            x = self.gnn_non_linear(x)
        prototype_activations, min_distances = self.prototype_distances(x)
        logits = self.last_layer(prototype_activations)
        probs = self.Softmax(logits)
        del prototype_activations
        return {"cf_pred":logits, "graph_pred": torch.stack([probs[:, 1], probs[:, 0]], dim=1), "graph_emb":x, "dist":min_distances}

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, min_dist: torch.Tensor
    ) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        pred_loss = criterion(logits, labels.logical_not().long())
        # cluster loss
        prototypes_of_correct_class = (
            self.prototype_class_identity[:, labels.logical_not().long()].bool().T
        )
        cluster_cost = (
            min_dist[prototypes_of_correct_class]
            .reshape(-1, self.prototypes_per_class)
            .min(dim=1)[0]
            .mean()
        )
        # seperation loss
        separation_cost = (
            -min_dist[~prototypes_of_correct_class]
            .reshape(-1, self.prototypes_per_class)
            .min(dim=1)[0]
            .mean()
        )
        # sparsity loss
        l1_mask = 1 - self.prototype_class_identity.T
        l1 = (self.last_layer.weight * l1_mask).norm(p=1)
        return pred_loss + 0.1 * cluster_cost + 0.05 * separation_cost + 5e-4 * l1


class MCTSNode:
    def __init__(
        self,
        coalition: list,
        c_puct: float = 10.0,
        W: float = 0,
        N: int = 0,
        P: float = 0,
    ):
        self.coalition = coalition
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)


def gnn_prot_score(
    coalition, data: Graph, gnn: GATNet, prototype: torch.Tensor, node_label: bool
):
    """the similarity value of subgraph with selected nodes"""
    mask = torch.zeros(data["x"].shape[0]).float()
    mask[coalition] = 1.0
    if len(data['x'].shape) > 1: 
        mask = mask.unsqueeze(1)
    model_output = gnn(
        (data["x"] * mask).unsqueeze(0).to(gnn.device),
        data["adj_matrix"].unsqueeze(0).to(gnn.device),
        data["edge_attr_matrix"].unsqueeze(0).to(gnn.device),
        node_label,
    )
    distance = torch.norm(model_output['graph_emb'] - prototype) ** 2
    return torch.log((distance + 1) / (distance + gnn.epsilon)).item()


def mcts_rollout(
    tree_node: MCTSNode,
    state_map: dict[str, MCTSNode],
    graph: nx.DiGraph,
    score_func: partial,
):
    if len(tree_node.coalition) <= 5:
        return tree_node.P

    # Expand if this node has never been visited
    if len(tree_node.children) == 0:
        degrees = list(graph.subgraph(tree_node.coalition).degree)
        nodes = [x[0] for x in sorted(degrees, key=lambda x: x[1])][:10]

        for n in nodes:
            # for each node, pruning it and get the remaining sub-graph
            # here we check the resulting sub-graphs and only keep the largest one
            sub_coalition = graph.subgraph([node for node in nodes if node != n])
            subgraphs = [
                graph.subgraph(c) for c in nx.connected_components(sub_coalition)
            ]
            new_coalition = sorted(list(max(subgraphs, key=len).nodes()))

            # check the state map and merge the same sub-graph
            new_node = MCTSNode(new_coalition)
            if str(new_coalition) not in state_map:
                state_map[str(new_coalition)] = new_node
            else:
                new_node = state_map[str(new_coalition)]
            if new_coalition not in [sorted(c.coalition) for c in tree_node.children]:
                tree_node.children.append(new_node)
        for c in tree_node.children:
            c.P = score_func(c.coalition) if c.P == 0 else c.P
    sum_count = sum([c.N for c in tree_node.children])
    selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, graph, score_func)
    selected_node.W += v
    selected_node.N += 1
    return v


def mcts(data, model, prototype, node_label):
    edge_index = data["adj_matrix"].nonzero().T
    graph = Data(
        x=data["x"][edge_index.unique()],
        edge_index=edge_index,
        edge_attr=data["edge_attr_matrix"][edge_index[0], edge_index[1]],
    )
    nx_graph = to_networkx(graph, to_undirected=True)
    root = MCTSNode([i for i in range(nx_graph.number_of_nodes())])
    state_map = {str(root.coalition): root}
    score_func = partial(
        gnn_prot_score, data=data, gnn=model, prototype=prototype, node_label=node_label
    )
    for _ in range(model.num_classes * model.prototypes_per_class):
        mcts_rollout(root, state_map, nx_graph, score_func)

    explanations = [node for _, node in state_map.items()]
    explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
    if len([e for e in explanations if len(e.coalition) <= 10]) > 0:
        result_node = next(e for e in explanations if len(e.coalition) <= 10)
    else:
        result_node = sorted(explanations, key=lambda x: len(x.coalition))[0]
    # compute the projected prototype to return
    mask = torch.zeros(data["x"].shape[0]).float()
    mask[result_node.coalition] = 1.0
    if len(data['x'].shape) > 1:
        mask = mask.unsqueeze(1)
    model_output = model(
        (data["x"] * mask).unsqueeze(0).to(model.device),
        data["adj_matrix"].unsqueeze(0).to(model.device),
        data["edge_attr_matrix"].unsqueeze(0).to(model.device),
        node_label,
    )
    return result_node.coalition, result_node.P, model_output['graph_emb'], model_output['cf_pred']


def prototype_projection(model: GATNet, data: GraphDataset):
    model.eval()
    for prot_idx in range(data.num_classes * model.prototypes_per_class):
        count, best_sim, proj_prot = 0, 0, 0
        while count < data.num_classes * model.prototypes_per_class:
            data_idx = random.sample(range(len(data)), 1)[0]
            if data[data_idx]["graph_label"] != prot_idx // model.prototypes_per_class:
                _, sim, proj, _ = mcts(
                    data[data_idx],
                    model,
                    model.prototype_vectors[prot_idx],
                    data.node_labels,
                )
                if sim > best_sim:
                    best_sim, proj_prot = sim, proj
                count += 1
        model.prototype_vectors.data[prot_idx] = proj_prot


def eval_gnn(eval_loader, gnn: GATNet):
    acc, loss = [], []
    gnn.eval()
    with torch.no_grad():
        for batch in eval_loader:
            model_output = gnn(
                batch["x"].to(gnn.device),
                batch["adj_matrix"].to(gnn.device),
                batch["edge_attr_matrix"].to(gnn.device),
                eval_loader.dataset.dataset.node_labels,
            )
            batch_loss = gnn.compute_loss(
                model_output['cf_pred'], batch["graph_label"].to(gnn.device), model_output['dist']
            )
            loss.append(batch_loss.item())
            acc.extend(
                model_output['cf_pred'].cpu().max(-1)[1].not_equal(batch["graph_label"]).float().tolist()
            )
    return np.average(loss), np.array(acc).mean()


def run_protgnn(dataset: GraphDataset, args: argparse.Namespace, model_save_path: str):
    results_path = model_save_path.replace("model_", "training-results_").replace(".pt", ".txt")
    with open(results_path, "w+") as f:
        f.write("Start training\n")
    train_loader, val_loader, test_loader = get_dataloaders(dataset, args.batch_size)
    best_loss, best_acc, early_stop_count = 1e10, 0, 0
    gnn = GATNet(
        dataset.num_node_features,
        args.embedding_dim,
        dataset.num_classes,
        dataset.num_edge_attributes,
        dataset.max_num_nodes,
        args.device,
        dataset.undesired_label,
        args.encoder_dropout
    ).to(args.device)
    if not os.path.exists(model_save_path):
        optimizer = torch.optim.Adam(gnn.parameters(), lr=args.gnn_lr)
        start = time.time()
        for epoch in range(args.gnn_epochs):
            if epoch >= args.gnn_epochs * 0.1 and epoch % 10 == 0:
                prototype_projection(gnn, dataset)
            gnn.train()
            for p in gnn.last_layer.parameters():
                p.requires_grad = epoch >= args.gnn_epochs * 0.1
            acc, loss = [], []
            for batch in train_loader:
                optimizer.zero_grad()
                model_output = gnn(
                    batch["x"].to(gnn.device),
                    batch["adj_matrix"].to(gnn.device),
                    batch["edge_attr_matrix"].to(gnn.device),
                    dataset.node_labels,
                )
                batch_loss = gnn.compute_loss(
                    model_output['cf_pred'], batch["graph_label"].to(args.device), model_output['dist']
                )
                batch_loss.backward()
                torch.nn.utils.clip_grad_value_(gnn.parameters(), clip_value=2.0)
                optimizer.step()
                loss.append(batch_loss.item())
                acc.extend(
                    model_output['cf_pred'].cpu().max(-1)[1].not_equal(batch["graph_label"]).float().tolist()
                )
            eval_loss, eval_acc = eval_gnn(val_loader, gnn)
            with open(results_path, "a+") as f:
                f.write(f"Epoch {epoch + 1}: loss = {np.average(loss):.3f} | acc = {np.array(acc).mean():.3f} | time = {time.time() - start}\n")
                f.write(f"Eval: loss = {eval_loss:.3f} | acc = {eval_acc:.3f}\n")
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}: loss = {np.average(loss):.3f} | acc = {np.array(acc).mean():.3f} | time = {time.time() - start}"
                )
                print(f"Eval: loss = {eval_loss:.3f} | acc = {eval_acc:.3f}")
            if eval_acc > best_acc:
                early_stop_count, best_acc, best_loss = 0, eval_acc, eval_loss
                torch.save(gnn.state_dict(), model_save_path)
                print(f"saved model: acc = {eval_acc}")
            elif epoch >= args.gnn_epochs * 0.1:
                early_stop_count += 1
            if early_stop_count > args.gnn_epochs * 0.75:
                break
    gnn.load_state_dict(torch.load(model_save_path, map_location=args.device))
    test_loss, test_acc = eval_gnn(test_loader, gnn)
    with open(results_path, "a+") as f: 
        f.write(f"Test: loss = {test_loss:.3f} | acc = {test_acc:.3f} | best_eval_loss = {best_loss:.3f} | best_eval_acc = {best_acc:.3f}\n")
    print(
        f"Test: loss = {test_loss:.3f} | acc = {test_acc:.3f} | best_eval_loss = {best_loss:.3f} | best_eval_acc = {best_acc:.3f}"
    )
    cfs = []
    for graph in tqdm(dataset):
        if graph['graph_label'] != dataset.undesired_label:
            continue
        best_nodes, best_sim = [], 0
        prototypes = gnn.prototype_vectors[
            gnn.prototype_class_identity[:, 1 - graph["graph_label"]].bool()
        ]
        for prot_idx in range(gnn.prototypes_per_class):
            nodes, sim, _, log = mcts(
                graph, gnn, prototypes[prot_idx], dataset.node_labels
            )
            if log.max(-1)[1] != graph["graph_label"] and sim > best_sim:
                best_nodes, best_sim = nodes, sim
        if len(best_nodes) > 0:
            mask = torch.zeros(graph["x"].shape[0]).float()
            mask[best_nodes] = 1.0
            if len(graph['x'].shape) > 1:
                mask = mask.unsqueeze(1)
            cfs.append(
                Graph(
                    x=graph["x"] * mask,
                    adj_matrix=graph["adj_matrix"],
                    edge_attr_matrix=graph["edge_attr_matrix"],
                    graph_idx=graph["graph_idx"],
                    graph_label=int(not graph["graph_label"]),
                )
            )
    return cfs, gnn
