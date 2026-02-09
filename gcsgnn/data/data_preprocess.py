import math
import os
import pickle
import random
import shutil
from importlib import import_module

import networkx as nx
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch_geometric.data.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from utils import ROOT_PATH


def generate_synthetic_data(
    max_num_nodes=12,
    min_num_nodes=8,
    num_node_feats=4,  # bottom, middle, top of house and not in house
    num_edge_feats=2,  # in motif and not in motif
    num_graphs=2000,
):
    dataset = []
    undesired_motif = nx.Graph()
    undesired_motif.add_nodes_from(
        [(0, {"x": 1}), (1, {"x": 2}), (2, {"x": 3}), (3, {"x": 2}), (4, {"x": 1})]
    )
    undesired_motif.add_edges_from(
        [
            (0, 1, {"edge_attr": 1}),
            (1, 2, {"edge_attr": 1}),
            (2, 3, {"edge_attr": 1}),
            (3, 4, {"edge_attr": 1}),
            (4, 0, {"edge_attr": 1}),
            (1, 3, {"edge_attr": 1}),
        ]
    )
    desired_motif = nx.Graph()
    desired_motif.add_nodes_from(
        [(0, {"x": 1}), (1, {"x": 2}), (2, {"x": 2}), (3, {"x": 1})]
    )
    desired_motif.add_edges_from(
        [
            (0, 1, {"edge_attr": 1}),
            (1, 2, {"edge_attr": 1}),
            (2, 3, {"edge_attr": 1}),
            (3, 0, {"edge_attr": 1}),
        ]
    )
    for label, (motif_graph_num, motif) in enumerate(
        [
            (math.ceil(num_graphs * 0.75), undesired_motif),
            (math.ceil(num_graphs * 0.25), desired_motif),
        ]
    ):
        for _ in tqdm(range(motif_graph_num)):
            cur_graph = motif.copy()
            num_nodes = random.randint(min_num_nodes, max_num_nodes)
            ba_graph = nx.barabasi_albert_graph(num_nodes - len(motif.nodes()), 1)
            ba_graph = nx.relabel_nodes(
                ba_graph, {n: n + len(motif.nodes()) for n in ba_graph.nodes()}
            )
            cur_graph.add_nodes_from(ba_graph.nodes())
            cur_graph.add_edges_from(ba_graph.edges())
            for node in ba_graph.nodes():
                cur_graph.nodes[node]["x"] = 0
            for edge in ba_graph.edges():
                cur_graph.edges[edge]["edge_attr"] = 0
            motif_node = random.sample(list(motif.nodes()), 1)[0]
            ba_node = random.sample(list(ba_graph.nodes()), 1)[0]
            cur_graph.add_edge(motif_node, ba_node, edge_attr=0)
            for _ in range(int(len(cur_graph.edges()) * 0.01)):
                while True:
                    u = random.randint(0, num_nodes)
                    v = random.randint(0, num_nodes)
                    if (not cur_graph.has_edge(u, v)) and (u != v):
                        break
                cur_graph.add_edge(u, v, edge_attr=0)
            node_order = torch.randperm(num_nodes)
            node_dict = {n: i for i, n in enumerate(node_order.tolist())}
            cur_graph = from_networkx(cur_graph)
            cur_graph.x = cur_graph.x[node_order]
            cur_graph.edge_index = torch.tensor(
                [[node_dict[val.item()] for val in row] for row in cur_graph.edge_index]
            )

            # cur_graph.edge_index = torch.stack([node_order[cur_graph.edge_index[i]] for i in range(2)], dim=0)
            cur_graph.edge_attr = cur_graph.edge_attr[cur_graph.edge_index.argsort()[1]]
            cur_graph.edge_index = cur_graph.edge_index[
                :, cur_graph.edge_index.argsort()[1]
            ]
            cur_graph.edge_attr = cur_graph.edge_attr[
                cur_graph.edge_index.argsort(stable=True)[0]
            ]
            cur_graph.edge_index = cur_graph.edge_index[
                :, cur_graph.edge_index.argsort(stable=True)[0]
            ]

            cur_graph.x = F.one_hot(cur_graph.x, num_classes=num_node_feats).float()
            cur_graph.edge_attr = F.one_hot(
                torch.tensor(cur_graph.edge_attr), num_classes=num_edge_feats
            ).float()
            cur_graph.y = torch.tensor(label)
            cur_graph.num_node_features = num_node_feats
            cur_graph.num_edge_features = num_edge_feats
            dataset.append(cur_graph)
    return dataset


def image_to_graph(img: tuple[Image.Image, int], size: int):
    transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    img_tensor = transform(img[0])
    G = nx.Graph()
    for i in range(size):
        for j in range(size):
            G.add_node((i, j))
    for i in range(size - 1):
        for j in range(1, size - 1):
            G.add_edge((i, j), (i, j + 1))
            G.add_edge((i, j), (i + 1, j - 1))
            G.add_edge((i, j), (i + 1, j))
            G.add_edge((i, j), (i + 1, j + 1))
        G.add_edge((i, 0), (i + 1, 0))
        G.add_edge((i, 0), (i + 1, 1))
    for j in range(size - 1):
        G.add_edge((size - 1, j), (size - 1, j + 1))
    adj = torch.tensor(nx.to_numpy_array(G)).float()
    x = img_tensor.flatten(-2).T
    return {"adj": adj, "x": x, "label": img[1]}


def get_raw_molecule_data(dataset_name: str) -> list[Data]:
    raw_dataset = TUDataset(root=f"{ROOT_PATH}datasets/", name=dataset_name)
    graphs = list(raw_dataset)
    shutil.rmtree(f"{ROOT_PATH}datasets/{dataset_name}/processed")
    return graphs


def get_raw_image_data(dataset_name: str) -> tuple[list[tuple[Image.Image, int]], int]:
    dataset_module = import_module("torchvision.datasets")
    dataset_class = getattr(dataset_module, dataset_name)
    raw_dataset = dataset_class(f"{ROOT_PATH}datasets/", download=True)
    return list(raw_dataset), max(raw_dataset.targets) + 1


def edit_molecule_data(graphs: list[Data], dataset_name: str) -> list[Data]:
    if dataset_name == "BZR_MD":
        for graph in graphs:
            graph.edge_index = graph.edge_index[:, graph.edge_attr.argmax(-1) != 1]
            new_edge_attr = torch.tensor(
                [e - 1 if e > 1 else e for e in graph.edge_attr.argmax(-1) if e != 1]
            )
            graph.edge_attr = F.one_hot(
                new_edge_attr, num_classes=graph.num_edge_features - 1
            ).float()
            graph.x = graph.x[graph.edge_index.flatten().unique(), :]
    for graph in graphs:
        graph.y = 1 if graph.y.item() == 1 else 0
    if graphs[0].x is not None:
        node_labels = torch.stack([graph.x.sum(-2) for graph in graphs])
        min_freq, new_graphs = len(graphs) * 0.01, []
        while len(new_graphs) < (len(graphs) * 0.01) and min_freq > (len(graphs) * 0.001):
            labels_to_keep = node_labels.sum(-2) > min_freq
            new_graphs = [
                graph
                for graph in graphs
                if graph.x.shape[0] == graph.x[:, labels_to_keep].sum()
            ]
            min_freq = 0.99 * min_freq
        for graph in new_graphs:
            graph.x = graph.x[:, labels_to_keep]
        graphs = new_graphs
    return graphs


def edit_image_data(
    images: list[tuple[Image.Image, int]], num_classes: int, size: int, dataset: str
) -> list[dict]:
    label_map = {
        label: [image for image in images if image[1] == label]
        for label in range(num_classes)
    }
    if dataset == "MNIST":
        class1, class2 = 0, 1
    elif dataset == "CIFAR10":
        class1, class2 = 0, 2
    random.shuffle(label_map[class1])
    random.shuffle(label_map[class2])
    images = (
        label_map[class1][: int(len(label_map[class1]) * 0.8)]
        + label_map[class2][: int(len(label_map[class2]) * 0.2)]
    )
    images = [(img[0], 0 if img[1] == class1 else 1) for img in images]
    if len(images) > 2000:
        random.shuffle(images)
        images = images[:2000]
    return [image_to_graph(img, size) for img in images]


def get_molecule_data_metadata(
    graphs: list[Data],
) -> tuple[int, int, int, int, int, int]:
    num_nodes = [graph.num_nodes for graph in graphs]
    # select the number of nodes of the top 1% largest graph
    max_num_nodes = sorted(num_nodes, reverse=True)[int(len(num_nodes) * 0.01)]
    num_node_features = (
        graphs[0].num_node_features + 1 if graphs[0].num_node_features != 0 else 2
    )
    num_edge_attributes = (
        graphs[0].num_edge_features + 1 if graphs[0].num_edge_features != 0 else 2
    )
    graph_labels = torch.tensor([graph.y for graph in graphs])
    unique_labels, label_count = graph_labels.unique(return_counts=True)
    num_classes, undesired_label = len(unique_labels), label_count.argmax().item()
    num_undesired_instances = label_count[undesired_label].item()
    return (
        max_num_nodes,
        num_node_features,
        num_edge_attributes,
        num_classes,
        undesired_label,
        num_undesired_instances,
    )


def get_image_data_metadata(images: list[dict]) -> tuple[int, int, int, int, int, int]:
    max_num_nodes = images[0]["adj"].shape[0]
    num_node_features, num_edge_attributes = images[0]["x"].shape[-1], 2
    graph_labels = torch.tensor([image["label"] for image in images])
    unique_labels, label_count = graph_labels.unique(return_counts=True)
    num_classes, undesired_label = (
        len(unique_labels),
        unique_labels[label_count.argmax()].item(),
    )
    num_undesired_instances = label_count.max().item()
    return (
        max_num_nodes,
        num_node_features,
        num_edge_attributes,
        num_classes,
        undesired_label,
        num_undesired_instances,
    )


def convert_molecule_data_to_lists(
    graphs: list[Data], max_num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    x_list, adj_list, edge_attr_list, label_list = [], [], [], []
    if len(graphs) > 2000:
        random.shuffle(graphs)
        graphs = graphs[:2000]
    for graph in graphs:
        if graph.num_nodes > max_num_nodes:
            continue
        adj = torch.zeros(max_num_nodes, max_num_nodes)
        adj[graph.edge_index[0], graph.edge_index[1]] = 1
        if graph.x is not None:
            x = F.pad(
                graph.x.argmax(-1) + 1, (0, max_num_nodes - graph.num_nodes)
            ).float()
        else:
            x = torch.zeros(max_num_nodes)
            x[graph.edge_index.unique()] = 1
        edge = adj.clone()
        if graph.edge_attr is not None:
            edge = torch.zeros(max_num_nodes, max_num_nodes)
            edge[graph.edge_index[0], graph.edge_index[1]] = (
                graph.edge_attr.argmax(-1) + 1
            ).float()
        (
            x_list.append(x),
            adj_list.append(adj),
            edge_attr_list.append(edge),
            label_list.append(graph.y),
        )
    return (
        torch.stack(x_list),
        torch.stack(adj_list),
        torch.stack(edge_attr_list),
        label_list,
    )


def convert_image_data_to_lists(
    graphs: list[Data],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    x_list = [graph["x"] for graph in graphs]
    adj_list = [graph["adj"] for graph in graphs]
    edge_attr_list = [graph["adj"] for graph in graphs]
    label_list = [graph["label"] for graph in graphs]
    return (
        torch.stack(x_list),
        torch.stack(adj_list),
        torch.stack(edge_attr_list),
        label_list,
    )


def save_graph_data(
    dataset_name: str,
    x_list: torch.Tensor,
    adj_list: torch.Tensor,
    edge_attr_list: torch.Tensor,
    label_list: list[int],
    max_num_nodes: int,
    num_node_features: int,
    num_edge_attributes: int,
    num_classes: int,
    undesired_label: int,
    num_undesired_instances: int,
    has_node_labels: bool,
) -> None:
    os.makedirs(f"{ROOT_PATH}datasets/{dataset_name}/processed")
    file_names = ["x", "adj", "label"]
    files = [x_list, adj_list, label_list]
    for i, file_name in enumerate(file_names):
        with open(
            f"{ROOT_PATH}datasets/{dataset_name}/processed/{file_name}.pkl", "wb"
        ) as f:
            pickle.dump(files[i], f, pickle.HIGHEST_PROTOCOL)
    with open(f"{ROOT_PATH}datasets/{dataset_name}/processed/edge_attr.pkl", "wb") as f:
        pickle.dump(edge_attr_list, f, pickle.HIGHEST_PROTOCOL)
    if os.path.exists(f"{ROOT_PATH}datasets/dataset_metadata.pkl"):
        with open(f"{ROOT_PATH}datasets/dataset_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        if dataset_name not in metadata:
            metadata[dataset_name] = {}
    else:
        metadata = {dataset_name: {}}
    metadata[dataset_name]["max_num_nodes"] = max_num_nodes
    metadata[dataset_name]["num_node_features"] = num_node_features
    metadata[dataset_name]["num_edge_attributes"] = num_edge_attributes
    metadata[dataset_name]["num_classes"] = num_classes
    metadata[dataset_name]["undesired_label"] = undesired_label
    metadata[dataset_name]["num_undesired"] = num_undesired_instances
    metadata[dataset_name]["node_labels"] = has_node_labels

    with open(f"{ROOT_PATH}datasets/dataset_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def process_raw_data(dataset_name, seed, size):
    if dataset_name in ["AIDS", "Mutagenicity", "BZR_MD", "MUTAG", "ENZYMES", "PTC_FM"]:
        graphs = get_raw_molecule_data(dataset_name)

        print("Processing data")
        random.Random(seed).shuffle(graphs)
        graphs = edit_molecule_data(graphs, dataset_name)
        (
            max_num_nodes,
            num_node_features,
            num_edge_attributes,
            num_classes,
            undesired_label,
            num_undesired_instances,
        ) = get_molecule_data_metadata(graphs)

        print("Converting data")
        x, adj, edge_attr, label = convert_molecule_data_to_lists(graphs, max_num_nodes)
        has_node_labels = True
    elif dataset_name == "synthetic":
        graphs = generate_synthetic_data()
        print("Processing data")
        random.Random(seed).shuffle(graphs)
        graphs = edit_molecule_data(graphs, dataset_name)
        (
            max_num_nodes,
            num_node_features,
            num_edge_attributes,
            num_classes,
            undesired_label,
            num_undesired_instances,
        ) = get_molecule_data_metadata(graphs)

        print("Converting data")
        x, adj, edge_attr, label = convert_molecule_data_to_lists(graphs, max_num_nodes)
        has_node_labels = True
    else:
        graphs, num_classes = get_raw_image_data(dataset_name)

        print("Processing data")
        random.Random(seed).shuffle(graphs)
        graphs = edit_image_data(graphs, num_classes, size, dataset_name)
        (
            max_num_nodes,
            num_node_features,
            num_edge_attributes,
            num_classes,
            undesired_label,
            num_undesired_instances,
        ) = get_image_data_metadata(graphs)

        print("Converting data")
        x, adj, edge_attr, label = convert_image_data_to_lists(graphs)
        has_node_labels = False
    print("Saving data")
    save_graph_data(
        dataset_name,
        x,
        adj,
        edge_attr,
        label,
        max_num_nodes,
        num_node_features,
        num_edge_attributes,
        num_classes,
        undesired_label,
        num_undesired_instances,
        has_node_labels,
    )
