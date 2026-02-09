import pickle
from typing import Optional, TypedDict

import torch
from torch.utils.data import Dataset
from utils import ROOT_PATH


class Graph(TypedDict):
    """
    Used to pass graphs to models (same as Graph_List but converts values to tensors)
    """
    x: torch.Tensor
    adj_matrix: torch.Tensor
    edge_attr_matrix: torch.Tensor
    graph_idx: int
    graph_label: int
    cf_idx: Optional[int]  # noqa: UP007


class GraphDataset(Dataset):
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        dataset_root_path = f"{ROOT_PATH}datasets/{dataset_name}/processed/"

        # reads processed data from .pkl files
        with open(f"{dataset_root_path}x.pkl", "rb") as f:
            x_list = pickle.load(f)
        with open(f"{dataset_root_path}adj.pkl", "rb") as f:
            adj_list = pickle.load(f)
        with open(f"{dataset_root_path}edge_attr.pkl", "rb") as f:
            edge_list = pickle.load(f)
        with open(f"{dataset_root_path}label.pkl", "rb") as f:
            label_list = pickle.load(f)
        self.graphs = []
        for i in range(x_list.shape[0]):
            graph = Graph(
                    x=x_list[i],
                    adj_matrix=adj_list[i],
                    edge_attr_matrix=edge_list[i],
                    graph_idx=i,
                    graph_label=label_list[i],
                )
            self.graphs.append(graph)

        # get max_num_nodes from dataset_metadata.pkl
        with open(f"{ROOT_PATH}datasets/dataset_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.max_num_nodes = metadata[dataset_name]["max_num_nodes"]
        self.num_node_features = metadata[dataset_name]["num_node_features"]
        self.num_edge_attributes = metadata[dataset_name]["num_edge_attributes"]
        self.num_classes = metadata[dataset_name]["num_classes"]
        self.undesired_label = metadata[dataset_name]["undesired_label"]
        self.num_undesired = metadata[dataset_name]["num_undesired"]
        self.node_labels = metadata[dataset_name]["node_labels"]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Graph:
        return self.graphs[idx]

    def get_graph_idx(self, graph_idx: int) -> Graph:
        for graph in self.graphs:
            if graph["graph_idx"] == graph_idx:
                return graph
        raise ValueError

    def filter_dataset(self, idxs_to_keep: list[int]) -> None:
        new_graph_list = [self.get_graph_idx(idx) for idx in idxs_to_keep]
        self.graphs = new_graph_list
        self.num_undesired = len(self.graphs)
