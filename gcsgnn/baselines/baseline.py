import types
from importlib import import_module

import torch
from data.data_utils import get_attribute_lists_from_data_list
from data.dataset import GraphDataset
from model.base_model import BaseModel


def get_baseline(baseline_type: str) -> types.FunctionType:
    module = import_module(f"baselines.{baseline_type}.{baseline_type}")
    baseline_method = getattr(module, "run_" + baseline_type)
    return baseline_method


def filter_dataset(dataset: GraphDataset, gnn: BaseModel, batch_size:int) -> GraphDataset:
    (
        all_x,
        all_adj,
        all_edge_attr,
        all_idxs,
        _,
    ) = get_attribute_lists_from_data_list(dataset.graphs, gnn.device)
    gnn_output = []
    for i in range((len(dataset) // batch_size) + ((len(dataset) % batch_size) != 0)):
        gnn_output.append(
            gnn(
                all_x[i * batch_size : (i + 1) * batch_size],
                all_adj[i * batch_size : (i + 1) * batch_size],
                all_edge_attr[i * batch_size : (i + 1) * batch_size],
                node_labels=dataset.node_labels
            )["graph_pred"].detach().cpu()
        )
    predictions = torch.argmax(torch.concat(gnn_output), dim=1).int()
    undesired_graphs_idxs = [
        all_idxs[i]
        for i, prediction in enumerate(predictions)
        if prediction == dataset.undesired_label
    ]
    dataset.filter_dataset(undesired_graphs_idxs)
    for i in range(len(dataset)):
        dataset.graphs[i]["graph_label"] = dataset.undesired_label
    all_x, all_adj, all_edge_attr, all_idxs = 0,0,0,0 
    torch.cuda.empty_cache()
    return dataset
