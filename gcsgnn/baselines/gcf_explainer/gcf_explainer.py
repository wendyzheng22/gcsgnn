import argparse
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines.baseline import filter_dataset
from data.dataset import Graph, GraphDataset
from model.base_model import BaseModel
from tqdm import tqdm


class GCFExplainer:
    def __init__(
        self,
        pred_model: nn.Module,
        theta: float,
        max_num_nodes: int,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        idx_list: list[int],
        device: str,
        undesired_label: int,
        batch_size: int,
        node_labels: bool,
    ) -> None:
        self.device = device
        self.alpha = 0.5
        self.sample_size = 10000
        self.max_num_cfs = 100000
        self.pred_model = pred_model
        self.theta = theta
        self.max_num_nodes = max_num_nodes
        self.x = x
        self.adj = adj
        self.edge_attr = edge_attr
        self.index = idx_list
        self.graphs_covered = torch.zeros(self.x.shape[0])
        self.hash_graph_map = {}
        self.cf_hashes = []
        self.undesired_label = undesired_label
        self.batch_size = batch_size
        self.node_labels = node_labels

    def get_hashes(
        self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor
    ) -> list[int]:
        embs = torch.cat([adj.flatten(-2), x.flatten(-2)], dim=-1)
        embs = torch.cat([embs, edge_attr.flatten(-3)], dim=-1)
        embs = torch.round(embs, decimals=2)
        hashes = [hash(emb.cpu().numpy().tobytes()) for emb in embs]
        return hashes

    def get_preds(
        self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        y_preds = []
        for i in range(
            (x.shape[0] // self.batch_size) + ((x.shape[0] % self.batch_size) != 0)
        ):
            cur_x = x[i * self.batch_size : (i + 1) * self.batch_size].to(self.device)
            cur_adj = adj[i * self.batch_size : (i + 1) * self.batch_size].to(
                self.device
            )
            cur_edge = edge_attr[i * self.batch_size : (i + 1) * self.batch_size].to(
                self.device
            )
            y_preds.append(
                self.pred_model(cur_x, cur_adj, cur_edge)["graph_pred"].detach().cpu()
            )
            del cur_x, cur_adj, cur_edge
        y_preds = torch.concat(y_preds, dim=0)
        return y_preds.detach().clone().exp()[:, int(not (self.undesired_label))]

    def get_distance(self, x_2: torch.Tensor, adj_2: torch.Tensor) -> torch.Tensor:
        distance = torch.zeros(x_2.shape[0], self.x.shape[0]).to(self.device)
        x, adj = self.x.to(self.device), self.adj.to(self.device)
        other_x, other_adj = x_2.to(self.device), adj_2.to(self.device)
        for i in range(other_x.shape[0]):
            dist_x = F.pairwise_distance(
                x.flatten(-2).unsqueeze(0), other_x[i].flatten(-2)
            )
            dist_x /= (
                x.argmax(-1).count_nonzero(dim=-1) if self.node_labels else x.shape[-2]
            )
            dist_a = (adj.flatten(-2) * other_adj[i].flatten(-2)).sum(-1) / adj.sum(
                (-1, -2)
            )
            distance[i] = (dist_x + dist_a) / 2
        return distance

    def get_graph_vals(
        self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        preds = self.get_preds(x, adj, edge_attr)
        preds = torch.stack([preds.cpu(), torch.ones(preds.shape)]).T
        hashes = self.get_hashes(x, adj, edge_attr)
        distances = self.get_distance(x, adj).squeeze()
        coverage_matrices = (distances <= self.theta).float().to_sparse().cpu()
        return preds, hashes, coverage_matrices

    def restart_random_walk(self):
        inv_exp_covered = 1 / self.graphs_covered.exp()
        dynamic_probs = inv_exp_covered / inv_exp_covered.sum()
        idx = random.choices(range(dynamic_probs.shape[0]), weights=dynamic_probs)[0]
        edge_attr = self.edge_attr[idx].unsqueeze(0)
        graph_pred, graph_hash, input_covering_list = self.get_graph_vals(
            self.x[idx].unsqueeze(0), self.adj[idx].unsqueeze(0), edge_attr
        )
        graph_hash = graph_hash[0]
        if graph_hash not in self.hash_graph_map:
            edge_attr = edge_attr.squeeze(0)
            self.hash_graph_map[graph_hash] = {
                "x": self.x[idx],
                "adj": self.adj[idx],
                "edge_attr": edge_attr,
                "imp_part": graph_pred,
                "covered_input_graphs": input_covering_list,
                "orig_graph_idx": self.index[idx],
            }
        return graph_hash

    def get_edge_actions(self, hash: int) -> list[tuple[str, int, int]]:
        actions = []
        adj = self.hash_graph_map[hash]["adj"]
        num_nodes = adj.sum(-1).nonzero()[-1].item() + 1
        for i, j in itertools.combinations(range(num_nodes), r=2):
            action = "remove_edge" if adj[i, j] else "add_edge"
            actions.append((action, i, j))
        return actions

    def edit_edge(
        self, adj: torch.Tensor, edge: torch.Tensor, action: tuple[str, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action[0] == "remove_edge":
            adj[action[1], action[2]], adj[action[1], action[2]] = 0, 0
            edge[action[1], action[2]] = torch.tensor([1] + [0] * (edge.shape[-1] - 1))
        elif action[0] == "add_edge":
            adj[action[1], action[2]], adj[action[1], action[2]] = 1, 1
            edge[action[1], action[2]] = torch.tensor(
                [0, 1] + [0] * (edge.shape[-1] - 2)
            )
        return adj, edge

    def get_node_actions(self, hash: int) -> list[tuple[str, int, int]]:
        x = self.hash_graph_map[hash]["x"]
        adj = self.hash_graph_map[hash]["adj"]
        num_nodes = (
            x.argmax(-1).nonzero()[-1].item() + 1 if self.node_labels else x.shape[0]
        )
        actions = []
        for i in range(num_nodes):
            if self.node_labels:
                actions.extend(
                    [
                        ("node_label", i, j)
                        for j in range(x.shape[-1])
                        if j != x.argmax(-1)[i]
                    ]
                )
            if num_nodes != x.shape[0]:
                actions.extend([("add_node", i, j) for j in range(x.shape[-1])])
            if adj.sum(-1)[i] == 1:
                actions.append(("remove_node", i, i))
        return actions

    def edit_node(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge: torch.Tensor,
        action: tuple[str, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action[0] == "node_label":
            x[action[1]] = 0
            x[action[1], action[2]] = 1
        elif action[0] == "add_node":
            num_nodes = x.argmax(-1).nonzero()[-1].item() + 1
            x[num_nodes] = 0
            x[num_nodes, action[2]] = 1
            adj[action[1], num_nodes], adj[num_nodes, action[1]] = 1, 1
            edge_attr = torch.tensor([0, 1] + [0] * (edge.shape[-1] - 2))
            edge[action[1], num_nodes, :] = edge_attr
        elif action[0] == "remove_node":
            zeros = torch.zeros(1, x.shape[-1])
            x = torch.concat([x[: action[1]], x[action[1] + 1 :], zeros])

            zeros = torch.zeros(1, x.shape[0])
            adj = torch.concat([adj[: action[1]], adj[action[1] + 1 :], zeros.clone()])
            adj = torch.concat(
                [adj[:, : action[1]], adj[:, action[1] + 1 :], zeros.clone().T], dim=1
            )

            e = edge.argmax(-1)
            e = torch.concat([e[: action[1]], e[action[1] + 1 :], zeros.clone()])
            e = torch.concat(
                [e[:, : action[1]], e[:, action[1] + 1 :], zeros.clone().T], dim=1
            )
            edge = F.one_hot(e.long(), num_classes=edge.shape[-1]).float()
        return x, adj, edge

    def get_neighbor_from_action(
        self, hash: int, action: tuple[str, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.hash_graph_map[hash]["x"].detach().clone()
        adj = self.hash_graph_map[hash]["adj"].detach().clone()
        edge = self.hash_graph_map[hash]["edge_attr"]
        edge = edge.detach().clone()
        if "edge" in action[0]:
            adj, edge = self.edit_edge(adj, edge, action)
        elif "node" in action[0]:
            x, adj, edge = self.edit_node(x, adj, edge, action)
        return x, adj, edge

    def find_neighbor_graphs(
        self, hash: int
    ) -> tuple[list[int], tuple[str, int, int], torch.Tensor, torch.Tensor]:
        actions = self.get_edge_actions(hash)
        actions.extend(self.get_node_actions(hash))
        if len(actions) > self.sample_size:
            selected_idxs = random.sample(range(len(actions)), self.sample_size)
            actions = [act for i, act in enumerate(actions) if i in selected_idxs]

        neighbors = [self.get_neighbor_from_action(hash, action) for action in actions]
        n_x, n_adj, n_edge = map(list, zip(*neighbors))
        n_x, n_adj = torch.stack(n_x), torch.stack(n_adj)
        n_edge = torch.stack(n_edge)

        neighbor_imp_parts, neighbor_hashes, neighbor_coverage = self.get_graph_vals(
            n_x, n_adj, n_edge
        )
        hashes = neighbor_hashes + [hash]
        imp_parts = torch.concat(
            [neighbor_imp_parts, self.hash_graph_map[hash]["imp_part"]]
        )
        actions = actions + [("NOTHING", None, None)]
        coverage = torch.concat(
            [
                neighbor_coverage,
                self.hash_graph_map[hash]["covered_input_graphs"].unsqueeze(0),
            ]
        ).to_dense()

        unique_idxs = np.unique(neighbor_hashes + [hash], return_index=True)[1].tolist()
        target_hashes = [hash for i, hash in enumerate(hashes) if i in unique_idxs]
        target_actions = [
            action for i, action in enumerate(actions) if i in unique_idxs
        ]
        target_imp_parts = imp_parts[unique_idxs, :]
        target_coverage_matrix = coverage[unique_idxs].to_sparse()
        return target_hashes, target_actions, target_imp_parts, target_coverage_matrix

    def calculate_importance(
        self, hashes: list[int], imp_parts: torch.Tensor, coverage: torch.Tensor
    ):
        belong = torch.tensor([hash in self.cf_hashes for hash in hashes])
        support = (
            coverage.to_dense()
            + (coverage.to_dense().T * belong).T
            - self.graphs_covered
        )
        cum_coverage = F.relu(support).sum(dim=1) / self.graphs_covered.shape[0]
        ind_coverage = coverage.to_dense().sum(dim=1) / self.graphs_covered.shape[0]
        imp_parts[:, 1] = self.alpha * ind_coverage + (1 - self.alpha) * cum_coverage
        imp_vals = (
            imp_parts[:, 0]
            if imp_parts[:, 1].sum() == 0
            else torch.prod(imp_parts, dim=1)
        )
        return imp_vals

    def get_next_graph_idx(self, hashes: list[int], imp_vals: torch.Tensor):
        freq = []
        for hash in hashes:
            if hash in self.hash_graph_map and "freq" in self.hash_graph_map[hash]:
                freq.append(self.hash_graph_map[hash]["freq"])
            elif len(self.cf_hashes) >= self.max_num_cfs:
                freq.append(self.hash_graph_map[self.cf_hashes[-1]]["freq"])
            else:
                freq.append(1)
        probs = imp_vals * torch.tensor(freq).float()
        probs = (
            probs / probs.sum()
            if probs.sum() != 0
            else torch.ones(probs.shape[0]) / probs.shape[0]
        )
        selected_hash_idx = random.choices(range(len(hashes)), weights=probs)[0]
        return selected_hash_idx

    def update_cfs(self, graph_hash: int):
        graph = self.hash_graph_map[graph_hash]
        if graph_hash in self.cf_hashes and graph["imp_part"][:, 0] >= 0.5:
            self.hash_graph_map[graph_hash]["freq"] += 1
        else:
            if len(self.cf_hashes) >= self.max_num_cfs:
                old_cf_hash = self.cf_hashes[-1]
                self.hash_graph_map[graph_hash]["freq"] = (
                    self.hash_graph_map[old_cf_hash] + 1
                )
                self.cf_hashes[-1] = graph_hash
                self.graphs_covered -= self.hash_graph_map[old_cf_hash][
                    "covered_input_graphs"
                ]
                del self.hash_graph_map[old_cf_hash]
            else:
                self.hash_graph_map[graph_hash]["freq"] = 2
                self.cf_hashes.append(graph_hash)
        freq = [self.hash_graph_map[hash]["freq"] for hash in self.cf_hashes]
        self.cf_hashes = [hash for _, hash in sorted(zip(freq, self.cf_hashes))]
        if graph["imp_part"].squeeze()[0] >= 0.5:
            self.graphs_covered += self.hash_graph_map[graph_hash][
                "covered_input_graphs"
            ]
        self.graphs_covered[self.graphs_covered < 0] = 0


def get_graph_data(
    dataset: GraphDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    x, adj, edge_attr, graph_idx = [], [], [], []
    for graph in dataset:
        cur_x = graph["x"].float()
        if dataset.node_labels:
            cur_x = F.one_hot(
                cur_x.long(), num_classes=dataset.num_node_features
            ).float()
        x.append(cur_x)
        adj.append(graph["adj_matrix"])
        edge_attr.append(
            F.one_hot(
                graph["edge_attr_matrix"].long(),
                num_classes=dataset.num_edge_attributes,
            ).float()
        )
        graph_idx.append(graph["graph_idx"])
    return torch.stack(x), torch.stack(adj), torch.stack(edge_attr), graph_idx


def run_gcf_explainer(
    pred_model: BaseModel,
    dataset: GraphDataset,
    args: argparse.Namespace,
    theta: float = 0.05,
) -> list[Graph]:
    pred_model.eval()
    dataset = filter_dataset(dataset, pred_model, args.batch_size)
    x, adj, edge_attr, idx_list = get_graph_data(dataset)
    explainer = GCFExplainer(
        pred_model,
        theta,
        dataset.max_num_nodes,
        x,
        adj,
        edge_attr,
        idx_list,
        args.device,
        dataset.undesired_label,
        args.batch_size,
        dataset.node_labels,
    )
    cur_hash = explainer.restart_random_walk()
    for _ in tqdm(range(10000)):
        explainer.update_cfs(cur_hash)
        if random.uniform(0, 1) < 0.1:
            cur_hash = explainer.restart_random_walk()
        else:
            if "transitions" not in explainer.hash_graph_map[cur_hash]:
                explainer.hash_graph_map[cur_hash]["transitions"] = (
                    explainer.find_neighbor_graphs(cur_hash)
                )
            neighbor_hashes, neighbor_actions, neighbor_imp_parts, neighbor_cov = (
                explainer.hash_graph_map[cur_hash]["transitions"]
            )
            imp_vals = explainer.calculate_importance(
                neighbor_hashes, neighbor_imp_parts, neighbor_cov
            )
            selected_idx = explainer.get_next_graph_idx(neighbor_hashes, imp_vals)
            if neighbor_hashes[selected_idx] not in explainer.hash_graph_map:
                n_x, n_a, n_e = explainer.get_neighbor_from_action(
                    cur_hash, neighbor_actions[selected_idx]
                )
                imp = neighbor_imp_parts[selected_idx].unsqueeze(0)
                cov = neighbor_cov[selected_idx]
                explainer.hash_graph_map[neighbor_hashes[selected_idx]] = {
                    "x": n_x,
                    "adj": n_a,
                    "edge_attr": n_e,
                    "imp_part": imp,
                    "covered_input_graphs": cov,
                    "orig_graph_idx": explainer.hash_graph_map[cur_hash][
                        "orig_graph_idx"
                    ],
                }
            cur_hash = neighbor_hashes[selected_idx]
    output = {}
    for hash in explainer.cf_hashes:
        if explainer.hash_graph_map[hash]["orig_graph_idx"] in output:
            output[explainer.hash_graph_map[hash]["orig_graph_idx"]].append(hash)
        else:
            output[explainer.hash_graph_map[hash]["orig_graph_idx"]] = [hash]
    final_hashes = []
    for orig_idx in output:
        pred = torch.stack(
            [
                explainer.hash_graph_map[hash]["imp_part"][0, 0]
                for hash in output[orig_idx]
            ]
        )
        final_hashes.append(output[orig_idx][pred.argmax()])
    final_output = []
    for hash in final_hashes:
        final_output.append(
            Graph(
                x=explainer.hash_graph_map[hash]["x"],
                adj_matrix=explainer.hash_graph_map[hash]["adj"],
                edge_attr_matrix=explainer.hash_graph_map[hash]["edge_attr"],
                graph_idx=explainer.hash_graph_map[hash]["orig_graph_idx"],
                graph_label=int(not bool(dataset.undesired_label)),
            )
        )
    return final_output
