import os

import torch
from data.data_preprocess import process_raw_data
from data.dataset import Graph, GraphDataset
from torch.utils.data import DataLoader, random_split
from utils import ROOT_PATH


def get_attribute_lists_from_data_list(
    graphs: list[Graph], device: str | None = None
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Separates the graph fields in a list of graphs into attribute lists
    """
    x_list = torch.stack([graph["x"] for graph in graphs], dim=0)
    adj_list = torch.stack([graph["adj_matrix"] for graph in graphs], dim=0)
    edge_attr_list = torch.stack([graph["edge_attr_matrix"] for graph in graphs], dim=0)
    graph_idxs_list = torch.tensor([graph["graph_idx"] for graph in graphs])
    graph_labels_list = torch.tensor([graph["graph_label"] for graph in graphs])
    if device is not None:
        x_list = x_list.to(device)
        adj_list = adj_list.to(device)
        edge_attr_list = edge_attr_list.to(device)
    return (
        x_list,
        adj_list,
        edge_attr_list,
        graph_idxs_list,
        graph_labels_list,
    )


def get_dataset(dataset_name: str, seed:int, size:int) -> GraphDataset:
    # process raw data if not done before and get dataset with processed data
    if not os.path.exists(f"{ROOT_PATH}datasets/{dataset_name}/processed"):
        process_raw_data(dataset_name, seed, size)
    dataset = GraphDataset(dataset_name)
    return dataset


def get_dataloaders(
    dataset: GraphDataset,
    batch_size: int,
    train_split: float = 0.5,
    test_split: float = 0.25,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # find number of data instances per split
    dataset_len = dataset.__len__()
    num_instances = [int(dataset_len * train_split), int(dataset_len * test_split)]
    num_instances.extend([dataset_len - (num_instances[0] + num_instances[1])])

    # get datasets
    train_dataset, test_dataset, val_dataset = random_split(dataset, num_instances)

    # get dataloaders
    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=batch_size, drop_last=len(train_dataset) % batch_size == 1
    )
    val_loader = DataLoader(
        dataset=val_dataset, shuffle=True, batch_size=batch_size, drop_last=len(val_dataset) % batch_size == 1
    )
    test_loader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=batch_size, drop_last=len(test_dataset) % batch_size == 1
    )

    return train_loader, val_loader, test_loader
